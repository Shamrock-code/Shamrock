// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CGLaplacianStencil.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief  7-point stencil for the discrete Laplacian operator
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
        const f64 *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {

        //        auto get_avg_neigh = [&](auto &graph_links) -> T {
        //     T acc   = shambase::VectorProperties<T>::get_zero();
        //     u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
        //         acc += field_access(id_b);
        //     });

        //     // T err_val = std::numeric_limits<T>::quiet_NaN();
        //     // return (cnt > 0) ? acc / cnt : err_val;
        //     return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        // };
        //         T W_i  = field_access(cell_global_id);
        // T W_xp = get_avg_neigh(graph_iter_xp);
        // T W_xm = get_avg_neigh(graph_iter_xm);
        // T W_yp = get_avg_neigh(graph_iter_yp);
        // T W_ym = get_avg_neigh(graph_iter_ym);
        // T W_zp = get_avg_neigh(graph_iter_zp);
        // T W_zm = get_avg_neigh(graph_iter_zm);

        // T inv_delta_cell_sqr = 1.0 / (delta_cell * delta_cell);

        // T laplace_x = inv_delta_cell_sqr * (-W_xm + 2. * W_i - W_xp);
        // T laplace_y = inv_delta_cell_sqr * (-W_ym + 2. * W_i - W_yp);
        // T laplace_z = inv_delta_cell_sqr * (-W_zm + 2. * W_i - W_zp);
        // T res       = (laplace_x + laplace_y + laplace_z);

        // return -res;

        auto cur_cell_block_id                    = cell_global_id / block_size;
        auto get_gradiant_dir_by_flux_formulation = [&](auto &graph_links, Direction dir) -> T {
            T acc                 = shambase::VectorProperties<T>::get_zero();
            auto cell_center_dist = cell_sizes[cur_cell_block_id];

            auto fac = 1.;
            u32 cnt  = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;

                int sign = 1 - 2 * (dir % 2);
                acc += sign * (field_access(id_b) - field_access(cell_global_id));

                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 2.);
                }
                // This logic suppose that the last (4-th) cell at interface have same size with the
                // other three cells. This is also consitent with 2:1 refinement.
                // TODO: extended to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
            });

            return (cnt > 0) ? acc / (cell_center_dist * fac * cnt)
                             : shambase::VectorProperties<T>::get_zero();
        };

        T flux_xp = get_gradiant_dir_by_flux_formulation(graph_iter_xp, Direction::xp);
        T flux_xm = get_gradiant_dir_by_flux_formulation(graph_iter_xm, Direction::xm);
        T flux_yp = get_gradiant_dir_by_flux_formulation(graph_iter_yp, Direction::yp);
        T flux_ym = get_gradiant_dir_by_flux_formulation(graph_iter_ym, Direction::ym);
        T flux_zp = get_gradiant_dir_by_flux_formulation(graph_iter_zp, Direction::zp);
        T flux_zm = get_gradiant_dir_by_flux_formulation(graph_iter_zm, Direction::zm);

        auto S = cell_sizes[cur_cell_block_id] * cell_sizes[cur_cell_block_id];

        //
        auto result = S * ((flux_xp + flux_yp + flux_zp) - (flux_xm + flux_ym + flux_zm));

        return -result; // the minus sign is for the positivity. We solve -\Delta \phi =
                        // -4*\pi*G*(\rho -\bar{rho})
    }
} // namespace shammodels::basegodunov
