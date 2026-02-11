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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @author Adnan-Ali Ahmad (adnan-ali.ahmad@cnrs.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)

 * @brief Computes the coordinates of each cell
 *
 */

#include "shammodels/ramses/modules/ComputeCoordinates.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeComputeCoordinates<Tvec, TgridVec>::_impl_evaluate_internal() {
        using Tscal    = shambase::VecComponent<Tvec>;
        using Config   = shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_cell0block_aabb_lower.check_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  u32 cell_count = block_count * block_size;
                  return cell_count;
              });

        edges.spans_coordinates.ensure_sizes(cell_counts);

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        auto patch_boxes = edges.patch_boxes.values;

        edges.sizes.indexes.for_each([&](u64 id, const u64 &n) { // for each patch
            sham::kernel_call( // loop through cells
                shamsys::instance::get_compute_scheduler().get_queue(),
                sham::MultiRef{edges.spans_block_cell_sizes.get_spans().get(id),
                                  edges.spans_cell0block_aabb_lower.get_spans().get(id)},
                sham::MultiRef{edges.spans_coordinates.get_spans().get(id)},
                cell_counts.get(id),
                [id, this, patch_boxes](
                    u64 i,
                    const Tscal *__restrict cell_size, // __restrict is there to prevent race conditions
                    const Tvec *__restrict cell0block_aabb_lower,
                    Tvec *__restrict coordinates) {

                    u32 block_id = i / block_size; // index of the block to which the current cell belongs
                    u32 cell_loc_id = i % block_size; // index of the cell within the block
                    Tvec pos_min_block = cell0block_aabb_lower[block_id]; // coordinates of the lower left corner of the block
                    Tscal cell_size_block = cell_size[block_id]; // size of the cells within the block

                    std::array<u32, dim> coor_array = AMRBlock::get_coord(cell_loc_id); // local coordinates of the cell within the block
                    // TODO make a loop to support any dimension
                    Tvec offset = Tvec{coor_array[0], coor_array[1], coor_array[2]} * cell_size_block; // offset of the cell center from the lower left corner of the block

                    shammath::AABB<TgridVec> patch_box = patch_boxes.get(id); // bounding box of the patch to which the block belongs

                    Tvec pos_min_patch = patch_box.lower.template convert<Tscal>() * grid_coord_to_pos_fact; // coordinates of the lower left corner of the patch

                    coordinates[i] =  pos_min_patch + pos_min_block + offset + 0.5 * cell_size_block; // coordinates of the cell center
                    }
                );
        });

        {
            auto &buffer_coordinates = edges.spans_coordinates.get_buf(0);
            u64 size                 = edges.spans_coordinates.get(0).get_obj_cnt();
            auto vec_coor            = buffer_coordinates.copy_to_stdvec();
            logger::raw("patch 0", size, "\n");
            for (int i = 0; i < size; i++) {
                logger::raw(vec_coor[i], "\n");
            }
        }

        {
            auto &buffer_coordinates = edges.spans_coordinates.get_buf(1);
            auto vec_coor            = buffer_coordinates.copy_to_stdvec();
            auto size                = edges.spans_coordinates.get(1).get_obj_cnt();

            logger::raw("patch 1", size, "\n");
            for (int i = 0; i < size; i++) {
                logger::raw(vec_coor[i], "\n");
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
