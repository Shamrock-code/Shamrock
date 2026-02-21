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
 * @author Adnan-Ali Ahmad (adnan-ali.ahmad@cnrs.fr) --no git blame--
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Noé Brucy (nbrucy@users.noreply.github.com)
 * @author Noé Brucy (noe.brucy@ens-lyon.fr) --no git blame--
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
        using Tscal = shambase::VecComponent<Tvec>;

        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size; // cell count
              });

        edges.spans_coordinates.ensure_sizes(cell_counts);
        auto &block_min_spans  = edges.spans_block_min.get_spans();
        auto &block_max_spans  = edges.spans_block_max.get_spans();
        auto &cell_coord_spans = edges.spans_coordinates.get_spans();

        Tscal one_over_Nside = 1. / block_nside;

        Tscal dxfact = grid_coord_to_pos_fact;

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{block_min_spans, block_max_spans},
            sham::DDMultiRef{cell_coord_spans},
            cell_counts,
            [one_over_Nside, dxfact](
                u32 i,
                const TgridVec *__restrict index_block_min,
                const TgridVec *__restrict index_block_max,
                Tvec *__restrict cell_coord) {
                u32 block_id = i / AMRBlock::block_size; // index of the block to which the current
                                                         // cell belongs
                u32 cell_loc_id = i % AMRBlock::block_size; // index of the cell within the block

                Tvec pos_block_min = index_block_min[block_id].template convert<Tscal>() * dxfact;
                Tvec pos_block_max = index_block_max[block_id].template convert<Tscal>() * dxfact;

                Tscal block_cell_size = ((pos_block_max - pos_block_min) * one_over_Nside).x();

                std::array<u32, dim> coord_array = AMRBlock::get_coord(cell_loc_id);

                Tvec offset; // offset of the lower left border of the cell from the lower left
                             // corner of the block
                for (u32 d = 0; d < dim; d++) {
                    offset[d] = coord_array[d] * block_cell_size;
                }

                // so the coordinate of the cell center is pos_block_min + offset + half a cell
                cell_coord[i] = pos_block_min + offset
                                + 0.5 * block_cell_size; // coordinates of the cell center
            });
    }

    template<class Tvec, class TgridVec>
    std::string NodeComputeCoordinates<Tvec, TgridVec>::_impl_get_tex() const {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto block_min   = get_ro_edge_base(1).get_tex_symbol();
        auto block_max   = get_ro_edge_base(2).get_tex_symbol();
        auto cell_coord  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute cell coordinates:

            \begin{align}
            s_i &= \mathbf{e}_x \cdot ({block_max}_i - {block_min}_i) \chi \\
            {cell_coord}_i &= \frac{s_i}{2} + {block_min}_i  \chi + \delta_{i \mod 8} \chi
            i &\in [0,{block_count} \cdot {block_nside}^3)\\
            \chi &= {grid_coord_to_pos_fact} / block_{\rm nside} \\
             \delta_{i \mod 8} &= \text{local cell offset within block} \\
            block_{\rm nside} &= {block_nside}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{cell_coord}", cell_coord);
        shambase::replace_all(tex, "{block_nside}", shambase::format("{}", block_nside));
        shambase::replace_all(
            tex, "{grid_coord_to_pos_fact}", shambase::format("{}", grid_coord_to_pos_fact));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeCoordinates<f64_3, i64_3>;
