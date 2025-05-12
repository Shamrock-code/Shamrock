// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCellInfos.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamdag/INode.hpp"
#include "shammodels/common/amr/AMRCellInfos.hpp"
#include "shammodels/ramses/modules/ComputeCellInfos.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/patch/dag_field.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"
#include <functional>
#include <memory>
#include <vector>

template<class Tvec, class TgridVec>
class Node_ComputeCellInfos : public INode {

    public:
    using Tscal    = shambase::VecComponent<Tvec>;
    using Config   = shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
    using AMRBlock = typename Config::AMRBlock;

    Tscal &scalingfactor;
    Node_ComputeCellInfos(Tscal &scalingfactor) : scalingfactor(scalingfactor) {}

    inline void set_inputs(
        std::shared_ptr<shamrock::dag::FieldRef<TgridVec, 1, shamrock::access_t_pointer>> block_min,
        std::shared_ptr<shamrock::dag::FieldRef<TgridVec, 1, shamrock::access_t_pointer>>
            block_max) {
        __internal_set_inputs({block_min, block_max});
    }
    inline void set_outputs(
        std::shared_ptr<
            shamrock::dag::FieldRef<Tscal, AMRBlock::block_size, shamrock::access_t_pointer>>
            cell_size,
        std::shared_ptr<
            shamrock::dag::FieldRef<Tvec, AMRBlock::block_size, shamrock::access_t_pointer>>
            cell_lower) {
        __internal_set_outputs({cell_size, cell_lower});
    }

    void _impl_evaluate_internal() {
        auto &block_min = get_input<shamrock::dag::FieldRef<TgridVec>>(0);
        auto &block_max = get_input<shamrock::dag::FieldRef<TgridVec>>(1);

        auto &block_cell_size = get_output<shamrock::dag::FieldRef<Tscal>>(0);
        auto &cell_lower      = get_output<shamrock::dag::FieldRef<Tvec>>(1);

        shambase::DistributedData<u32> block_sizes
            = block_min.field_refs.template map<u32>([](u64, auto &buf) {
                  return buf.count;
              });

        Tscal one_over_Nside = 1. / AMRBlock::Nside;

        Tscal dxfact = scalingfactor;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{block_min, block_max},
            sham::DDMultiRef{block_cell_size, cell_lower},
            block_sizes,
            [one_over_Nside, dxfact](
                u32 i,
                const TgridVec *block_min,
                const TgridVec *block_max,
                Tscal *cell_size,
                Tvec *cell_lower) {
                TgridVec lower = block_min[i];
                TgridVec upper = block_max[i];

                Tvec lower_flt = lower.template convert<Tscal>() * dxfact;
                Tvec upper_flt = upper.template convert<Tscal>() * dxfact;

                Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

                Tscal res = block_cell_size.x();

                cell_size[i]  = res;
                cell_lower[i] = lower_flt;
            });
    }

    void _impl_reset_internal() {}

    inline std::string _impl_get_label() { return "Node Compute Cell Infos"; }
    inline std::string _impl_get_node_tex() {

        std::string block_min
            = get_input<shamrock::dag::FieldRef<TgridVec>>(0).get_tex_symbol() + "_a";
        std::string block_max
            = get_input<shamrock::dag::FieldRef<TgridVec>>(1).get_tex_symbol() + "_a";

        std::string cell_size
            = get_output<shamrock::dag::FieldRef<Tscal>>(0).get_tex_symbol() + "_a";
        std::string cell_lower
            = get_output<shamrock::dag::FieldRef<Tvec>>(1).get_tex_symbol() + "_a";

        return ""; // TODO "\\[" + rho + " = \\frac{" + mass + "}{" + h + "}\\]";
    }
};

class DAGInput : public INode {

    public:
    DAGInput() {}

    inline void set_inputs() { __internal_set_inputs({}); }
    inline void set_outputs(std::vector<std::shared_ptr<IDataEdge>> out) {
        __internal_set_outputs({out});
    }

    void _impl_evaluate_internal() {}

    void _impl_reset_internal() {}

    inline std::string _impl_get_label() { return "DAG input"; }
    inline std::string _impl_get_node_tex() { return ""; }
};

class DAGOutput : public INode {

    public:
    DAGOutput() {}

    inline void set_inputs(std::vector<std::shared_ptr<IDataEdge>> in) {
        __internal_set_inputs({in});
    }
    inline void set_outputs() { __internal_set_outputs({}); }

    void _impl_evaluate_internal() {}

    void _impl_reset_internal() {}

    inline std::string _impl_get_label() { return "DAG input"; }
    inline std::string _impl_get_node_tex() { return ""; }
};

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeCellInfos<Tvec, TgridVec>::compute_aabb() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    logger::debug_ln("AMR grid", "compute block/cell infos");

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tscal> block_cell_sizes
        = utility.make_compute_field<Tscal>("aabb cell size", 1, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    shamrock::ComputeField<Tvec> cell0block_aabb_lower
        = utility.make_compute_field<Tvec>("aabb cell lower", 1, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::EventList depends_list;

        auto acc_block_min = buf_block_min.get_read_access(depends_list);
        auto acc_block_max = buf_block_max.get_read_access(depends_list);
        auto bsize         = block_cell_sizes.get_buf(id).get_write_access(depends_list);
        auto aabb_lower    = cell0block_aabb_lower.get_buf(id).get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal one_over_Nside = 1. / AMRBlock::Nside;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            shambase::parralel_for(cgh, mpdat.total_elements, "compute cell infos", [=](u32 gid) {
                TgridVec lower = acc_block_min[gid];
                TgridVec upper = acc_block_max[gid];

                Tvec lower_flt = lower.template convert<Tscal>() * dxfact;
                Tvec upper_flt = upper.template convert<Tscal>() * dxfact;

                Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

                Tscal res = block_cell_size.x();

                bsize[gid]      = res;
                aabb_lower[gid] = lower_flt;
            });
        });

        buf_block_min.complete_event_state(e);
        buf_block_max.complete_event_state(e);
        block_cell_sizes.get_buf(id).complete_event_state(e);
        cell0block_aabb_lower.get_buf(id).complete_event_state(e);
    });

    storage.cell_infos.set(
        CellInfos<Tvec, TgridVec>{std::move(block_cell_sizes), std::move(cell0block_aabb_lower)});

    std::shared_ptr<shamrock::dag::FieldRef<TgridVec>> block_min
        = std::make_shared<shamrock::dag::FieldRef<TgridVec>>("block_min", "\\mathbf{b}_{min}");
    std::shared_ptr<shamrock::dag::FieldRef<TgridVec>> block_max
        = std::make_shared<shamrock::dag::FieldRef<TgridVec>>("block_max", "\\mathbf{b}_{max}");

    std::shared_ptr<shamrock::dag::FieldRef<Tscal>> cell_size
        = std::make_shared<shamrock::dag::FieldRef<Tscal>>("cell_size", "\\mathbf{c}_{size}");
    std::shared_ptr<shamrock::dag::FieldRef<Tvec>> cell_aabb_lower
        = std::make_shared<shamrock::dag::FieldRef<Tvec>>("cell_aabb_lower", "\\mathbf{c}_{lower}");

    Node_ComputeCellInfos<Tvec, TgridVec> node(solver_config.grid_coord_to_pos_fact);
    node.set_inputs(block_min, block_max);
    node.set_outputs(cell_size, cell_aabb_lower);

    NodeAttachToFields<TgridVec> attach_block_min(
        storage.merged_patchdata_ghost.get()
            .template map<std::reference_wrapper<PatchDataField<TgridVec>>>(
                [&](u64 id, MergedPDat &mpdat) -> std::reference_wrapper<PatchDataField<TgridVec>> {
                    return mpdat.pdat.get_field<TgridVec>(0);
                }));

    NodeAttachToFields<TgridVec> attach_block_max(
        storage.merged_patchdata_ghost.get()
            .template map<std::reference_wrapper<PatchDataField<TgridVec>>>(
                [&](u64 id, MergedPDat &mpdat) -> std::reference_wrapper<PatchDataField<TgridVec>> {
                    return mpdat.pdat.get_field<TgridVec>(1);
                }));

    attach_block_min.set_outputs(block_min);
    attach_block_max.set_outputs(block_max);

    std::shared_ptr<NodeStoreComputeField<Tscal>> store_cell_size
}

template class shammodels::basegodunov::modules::ComputeCellInfos<f64_3, i64_3>;
