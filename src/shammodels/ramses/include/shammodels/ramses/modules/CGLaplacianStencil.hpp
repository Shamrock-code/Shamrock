// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CGLaplacianStencil.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
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
    template<class T, class Tvec, class ACCField>
    inline T laplacian_7pt(
        const u32 cell_global_id,
        const T delta_cell,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {
        u32 g_cnt          = 0;
        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            // g_cnt += cnt;
            // logger::raw(cnt, g_cnt);
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        T W_i                = field_access(cell_global_id);
        T W_xp               = get_avg_neigh(graph_iter_xp);
        T W_xm               = get_avg_neigh(graph_iter_xm);
        T W_yp               = get_avg_neigh(graph_iter_yp);
        T W_ym               = get_avg_neigh(graph_iter_ym);
        T W_zp               = get_avg_neigh(graph_iter_zp);
        T W_zm               = get_avg_neigh(graph_iter_zm);
        T inv_delta_cell_sqr = 1.0 / (delta_cell * delta_cell);
        // logger::raw("global", g_cnt);

        // logger::raw_ln(cell_global_id, W_xm, W_ym, W_zm, W_xp, W_yp, W_zp, "\n");

        //  logger::raw_ln(W_i);

        // logger::raw_ln(inv_delta_cell_sqr);

        T laplace_x = inv_delta_cell_sqr * (-W_xm + 2. * W_i - W_xp);
        T laplace_y = inv_delta_cell_sqr * (-W_ym + 2. * W_i - W_yp);
        T laplace_z = inv_delta_cell_sqr * (-W_zm + 2. * W_i - W_zp);

        T res = 0;

        // if(0 <= cell_global_id && cell_global_id <= 63 )
        res = (laplace_x + laplace_y + laplace_z);
        // logger::raw_ln(res);

        return res;
    }
} // namespace shammodels::basegodunov
