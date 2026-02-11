// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCoordinates.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeCoordinates.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec, class TgridVec>
    struct KernelComputeCoordinates {
        using Tscal    = shambase::VecComponent<Tvec>;
        using Config   = shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        inline static void kernel(
            u32 block_size,
            Tscal grid_coord_to_pos_fact,
            const shambase::DistributedData<u32>
                cell_counts, // number of cells per patch for all patches on the current MPI process
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_block_cell_sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<u64>> &patch_ids,
            const shambase::DistributedData<shammath::AABB<TgridVec>> &patch_boxes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_cell0block_aabb_lower,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_coordinates) {

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{patch_ids, spans_block_cell_sizes, spans_cell0block_aabb_lower},
                sham::DDMultiRef{spans_coordinates},
                cell_counts,
                [block_size, grid_coord_to_pos_fact, patch_boxes](
                    u32 i,
                    const u64 *__restrict patch_id,
                    const Tscal
                        *__restrict cell_size, // __restrict is there to prevent race conditions
                    const Tvec *__restrict cell0block_aabb_lower,
                    Tvec *__restrict coordinates) {
                    u32 block_id
                        = i / block_size; // index of the block to which the current cell belongs
                    u32 cell_loc_id = i % block_size; // index of the cell within the block
                    Tvec pos_min_block
                        = cell0block_aabb_lower[block_id]; // coordinates of the lower left corner
                                                           // of the block
                    Tscal cell_size_block
                        = cell_size[block_id]; // size of the cells within the block

                    std::array<u32, dim> coor_array = AMRBlock::get_coord(
                        cell_loc_id); // local coordinates of the cell within the block
                    // TODO make a loop to support any dimension
                    Tvec offset = Tvec{coor_array[0], coor_array[1], coor_array[2]}
                                  * cell_size_block; // offset of the cell center from the lower
                                                     // left corner of the block

                    shammath::AABB<TgridVec> patch_box = patch_boxes.get(
                        patch_id[block_id]); // bounding box of the patch to which the block belongs

                    Tvec pos_min_patch = patch_box.lower.template convert<Tscal>()
                                         * grid_coord_to_pos_fact; // coordinates of the lower left
                                                                   // corner of the patch

                    coordinates[i] = pos_min_patch + pos_min_block + offset
                                     + 0.5 * cell_size_block; // coordinates of the cell center
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeComputeCoordinates<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_cell0block_aabb_lower.check_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        shambase::DistributedData<PatchDataField<u64>> patch_id_dd;
        edges.sizes.indexes.for_each([&](u64 id, u32 block_count) {
            patch_id_dd.add_obj(id, std::move(PatchDataField<u64>("id_patch", 1, block_count)));
        });

        shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<u64>> patch_ids
            = edges.sizes.indexes.template map<shamrock::PatchDataFieldSpanPointer<u64>>(
                [&](u64 id, u32 block_count) {
                    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
                    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();
                    sham::kernel_call(
                        q,
                        sham::MultiRef{},
                        sham::MultiRef{patch_id_dd.get(id).get_buf()},
                        block_count,
                        [id](u32 i, u64 *__restrict patch_id) {
                            patch_id[i] = id;
                        });
                    return patch_id_dd.get(id).get_pointer_span();
                });

        edges.spans_coordinates.ensure_sizes(cell_counts);

        KernelComputeCoordinates<Tvec, TgridVec>::kernel(
            block_size,
            grid_coord_to_pos_fact,
            cell_counts,
            edges.spans_block_cell_sizes.get_spans(),
            patch_ids,
            edges.patch_boxes.values,
            edges.spans_cell0block_aabb_lower.get_spans(),
            edges.spans_coordinates.get_spans());

        {
            auto &buffer_coordinates = edges.spans_coordinates.get_buf(0);
            auto vec_coor            = buffer_coordinates.copy_to_stdvec();
            logger::raw(buffer_coordinates.get_size(), "\n");
            for (int i = 0; i < buffer_coordinates.get_size(); i++) {
                logger::raw(vec_coor[i]);
            }
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeComputeCoordinates<Tvec, TgridVec>::_impl_get_tex() const {

        std::string tex = R"tex(
            Compute cell coordinates:

            \begin{align*}
                \bm{r} &=  \text{coor_patch} + \text{coor_block} + \text{cell\_size} * (\text{get\_coord}(i) + 0.5 )
            \end{align*}
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeCoordinates<f64_3, i64_3>;
