// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CGLaplacianStencil.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include <shambackends/sycl.hpp>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
namespace shammodels::basegodunov {
    using Direction = shammodels::basegodunov::modules::Direction;

    /**
     * @brief Get the discretized laplacian
     *
     * @tparam T
     * @tparam Tvec
     * @tparam ACCField
     * @param cell_global_id
     * @param delta_cell
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access
     * @return T
     */
    template<
        class T,
        class Tvec,
        class TUint,
        class ACCField1,
        class ACCField2,
        class ACCField3,
        class ACCField4>
    inline T laplacian_7pt(
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField1 &&field_access,
        ACCField2 &&block_level_acc,
        ACCField3 &&block_min_acc,
        ACCField4 &&block_max_acc) {

        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        auto get_neigh_AMRlevel = [&](auto &graph_links) {
            auto lev = (TUint) 0;
            u32 cnt  = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                lev = block_level_acc(id_b);
            });
            return lev;
        };

        auto bloc_size = block_max_acc(cell_global_id) - block_min_acc(cell_global_id);

        auto lev_diff_xp = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_xp);
        auto lev_diff_xm = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_xm);
        auto lev_diff_yp = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_yp);
        auto lev_diff_ym = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_ym);
        auto lev_diff_zp = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_zp);
        auto lev_diff_zm = block_level_acc(cell_global_id) - get_neigh_AMRlevel(graph_iter_zm);

        T W_i  = field_access(cell_global_id);
        T W_xp = get_avg_neigh(graph_iter_xp);
        T W_xm = get_avg_neigh(graph_iter_xm);
        T W_yp = get_avg_neigh(graph_iter_yp);
        T W_ym = get_avg_neigh(graph_iter_ym);
        T W_zp = get_avg_neigh(graph_iter_zp);
        T W_zm = get_avg_neigh(graph_iter_zm);

        T dx = bloc_size.x();
        T dy = bloc_size.y();
        T dz = bloc_size.z();

        T laplace_x
            = (W_i - W_xm)
                  * (1.0 / ((lev_diff_xm > 0) ? dx * 0.75 : ((lev_diff_xm < 0) ? dx * 1.5 : dx)))
              + (W_xp - W_i)
                    * (1.0 / ((lev_diff_xp > 0) ? dx * 0.75 : ((lev_diff_xp < 0) ? dx * 1.5 : dx)));

        T laplace_y
            = (W_i - W_ym)
                  * (1.0 / ((lev_diff_ym > 0) ? dy * 0.75 : ((lev_diff_ym < 0) ? dy * 1.5 : dy)))
              + (W_yp - W_i)
                    * (1.0 / ((lev_diff_yp > 0) ? dy * 0.75 : ((lev_diff_yp < 0) ? dy * 1.5 : dy)));

        T laplace_z
            = (W_i - W_zm)
                  * (1.0 / ((lev_diff_zm > 0) ? dz * 0.75 : ((lev_diff_zm < 0) ? dz * 1.5 : dz)))
              + (W_zp - W_i)
                    * (1.0 / ((lev_diff_zp > 0) ? dz * 0.75 : ((lev_diff_zp < 0) ? dz * 1.5 : dz)));

        T res = -(laplace_x * 1.0 / dx) - (laplace_y * 1.0 / dy) - (laplace_z * 1.0 / dz);

        return res;
    }
} // namespace
