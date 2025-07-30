// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCellAABB.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeCellAABB.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec, class TgridVec>
    struct KernelComputeCellAABB {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<TgridVec>>
                &spans_block_min,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<TgridVec>>
                &spans_block_max,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_block_cell_sizes,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_cell0block_aabb_lower,
            const shambase::DistributedData<u32> &sizes,
            u32 block_nside,
            Tscal grid_coord_to_pos_fact) {

            const shambase::DistributedData<u32> &block_counts = sizes;

            Tscal one_over_Nside = 1. / block_nside;

            Tscal dxfact = grid_coord_to_pos_fact;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_block_min, spans_block_max},
                sham::DDMultiRef{spans_block_cell_sizes, spans_cell0block_aabb_lower},
                block_counts,
                [one_over_Nside, dxfact](
                    u32 i,
                    const TgridVec *__restrict acc_block_min,
                    const TgridVec *__restrict acc_block_max,
                    Tscal *__restrict bsize,
                    Tvec *__restrict aabb_lower) {
                    TgridVec lower = acc_block_min[i];
                    TgridVec upper = acc_block_max[i];

                    Tvec lower_flt = lower.template convert<Tscal>() * dxfact;
                    Tvec upper_flt = upper.template convert<Tscal>() * dxfact;

                    Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

                    Tscal res = block_cell_size.x();

                    bsize[i]      = res;
                    aabb_lower[i] = lower_flt;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeComputeCellAABB<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        edges.spans_block_cell_sizes.ensure_sizes(edges.sizes.indexes);
        edges.spans_cell0block_aabb_lower.ensure_sizes(edges.sizes.indexes);

        KernelComputeCellAABB<Tvec, TgridVec>::kernel(
            edges.spans_block_min.get_spans(),
            edges.spans_block_max.get_spans(),
            edges.spans_block_cell_sizes.get_spans(),
            edges.spans_cell0block_aabb_lower.get_spans(),
            edges.sizes.indexes,
            block_nside,
            grid_coord_to_pos_fact);
    }

    template<class Tvec, class TgridVec>
    std::string NodeComputeCellAABB<Tvec, TgridVec>::_impl_get_tex() {

        auto block_count           = get_ro_edge_base(0).get_tex_symbol();
        auto block_min             = get_ro_edge_base(1).get_tex_symbol();
        auto block_max             = get_ro_edge_base(2).get_tex_symbol();
        auto block_cell_sizes      = get_rw_edge_base(0).get_tex_symbol();
        auto cell0block_aabb_lower = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Compute cell AABBs

            \begin{align}
            {block_cell_sizes}_i &= \mathbf{e}_x \cdot ({block_max}_i - {block_min}_i) (\chi / block_{\rm nside}) \\
            {cell0block_aabb_lower}_i &= ({block_min}_i) (\chi / block_{\rm nside}) \\
            i &\in [0,{block_count})\\
            \chi &= {grid_coord_to_pos_fact}\\
            block_{\rm nside} &= {block_nside}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{block_cell_sizes}", block_cell_sizes);
        shambase::replace_all(tex, "{cell0block_aabb_lower}", cell0block_aabb_lower);
        shambase::replace_all(tex, "{block_nside}", shambase::format("{}", block_nside));
        shambase::replace_all(
            tex, "{grid_coord_to_pos_fact}", shambase::format("{}", grid_coord_to_pos_fact));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeCellAABB<f64_3, i64_3>;
