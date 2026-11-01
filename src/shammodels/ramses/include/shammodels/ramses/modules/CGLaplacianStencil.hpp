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
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include <shambackends/sycl.hpp>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
namespace shammodels::basegodunov {
    using Direction = shammodels::basegodunov::modules::Direction;

    /**
     * @brief Get the discretized (finite volume) Laplacian, and the diagonal coefficient of the
     * associated matrix.
     *
     * The Laplacian is written in flux form: for each direction the gradient across the
     * interface is the mean over the neighbours of \f$ (W_b - W_i) / d \f$, \f$ d \f$ being the
     * distance between the cell centers (\f$ \Delta x \f$ for a neighbour of the same level,
     * \f$ 3 \Delta x / 2 \f$ for a coarser one and \f$ 3 \Delta x / 4 \f$ for a finer one, assuming
     * a 2:1 refinement). The fluxes are then multiplied by the face surface
     * \f$ \Delta x^2 \f$, so that the result is \f$ -\Delta x^3 \Delta W \f$ (the volume integral
     * of \f$ -\Delta W \f$ over the cell). A direction without any neighbour contributes a zero
     * flux.
     *
     * The matrix is symmetric, including across level jumps (the coarse/fine coupling is
     * \f$ \Delta x_{\rm coarse} / 3 \f$ from both sides). The scheme is second order on a uniform
     * grid, but at a level jump the two-point flux ignores the lateral offset between the cell
     * centers: the pointwise truncation error is O(1) there and the scheme is first order in the
     * weak sense (see NodeSpMVPoisson3DTests.cpp).
     *
     * The minus sign makes the matrix positive. With it we solve
     * \f$ -\Delta \phi = -4 \pi G (\rho - \bar{\rho}) \f$ (multiplied by the cell volume).
     *
     * @tparam T field type
     * @tparam Tvec vector type
     * @tparam ACCField field accessor type, `T(u32 cell_id)`
     * @param cell_sizes cell size per block
     * @param block_size number of cells per block
     * @param cell_global_id id of the cell
     * @param graph_iter_xp neighbour graph accessor in the +x direction
     * @param graph_iter_xm neighbour graph accessor in the -x direction
     * @param graph_iter_yp neighbour graph accessor in the +y direction
     * @param graph_iter_ym neighbour graph accessor in the -y direction
     * @param graph_iter_zp neighbour graph accessor in the +z direction
     * @param graph_iter_zm neighbour graph accessor in the -z direction
     * @param field_access field accessor
     * @param Aii diagonal coefficient of the matrix for this cell (output)
     * @return T the discretized \f$ -\Delta x^3 \Delta W \f$
     */
    template<class T, class Tvec, class ACCField>
    inline T laplacian_7pt_with_Aii(
        const shambase::VecComponent<Tvec> *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access,
        T &Aii) {

        auto cur_cell_block_id = cell_global_id / block_size;
        Aii                    = shambase::VectorProperties<T>::get_zero();

        auto get_gradient_dir_by_flux_formulation = [&](auto &graph_links, Direction dir) -> T {
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
                // This logic supposes that the last (4-th) cell at the interface has the same size
                // as the other three cells. This is consistent with 2:1 refinement.
                // TODO: extend to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
            });

            // TODO: (kept from #1942) this is added even when cnt == 0, whereas the flux of a
            // direction without neighbours is zero, so Aii is then larger than the actual diagonal
            // coefficient of the matrix (only matters for non periodic boundaries).
            Aii += 1. / (cell_center_dist * fac);
            return (cnt > 0) ? acc / (cell_center_dist * fac * cnt)
                             : shambase::VectorProperties<T>::get_zero();
        };

        T flux_xp = get_gradient_dir_by_flux_formulation(graph_iter_xp, Direction::xp);
        T flux_xm = get_gradient_dir_by_flux_formulation(graph_iter_xm, Direction::xm);
        T flux_yp = get_gradient_dir_by_flux_formulation(graph_iter_yp, Direction::yp);
        T flux_ym = get_gradient_dir_by_flux_formulation(graph_iter_ym, Direction::ym);
        T flux_zp = get_gradient_dir_by_flux_formulation(graph_iter_zp, Direction::zp);
        T flux_zm = get_gradient_dir_by_flux_formulation(graph_iter_zm, Direction::zm);

        auto S = cell_sizes[cur_cell_block_id] * cell_sizes[cur_cell_block_id];

        auto result = S * ((flux_xp + flux_yp + flux_zp) - (flux_xm + flux_ym + flux_zm));
        Aii *= S;

        return -result;
    }

    /**
     * @brief Get the discretized (finite volume) Laplacian
     *
     * Same as laplacian_7pt_with_Aii (see it for the details of the discretization), without the
     * diagonal coefficient.
     *
     * @tparam T field type
     * @tparam Tvec vector type
     * @tparam ACCField field accessor type, `T(u32 cell_id)`
     * @param cell_sizes cell size per block
     * @param block_size number of cells per block
     * @param cell_global_id id of the cell
     * @param graph_iter_xp neighbour graph accessor in the +x direction
     * @param graph_iter_xm neighbour graph accessor in the -x direction
     * @param graph_iter_yp neighbour graph accessor in the +y direction
     * @param graph_iter_ym neighbour graph accessor in the -y direction
     * @param graph_iter_zp neighbour graph accessor in the +z direction
     * @param graph_iter_zm neighbour graph accessor in the -z direction
     * @param field_access field accessor
     * @return T the discretized \f$ -\Delta x^3 \Delta W \f$
     */
    template<class T, class Tvec, class ACCField>
    inline T laplacian_7pt(
        const shambase::VecComponent<Tvec> *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {
        T Aii_unused;
        return laplacian_7pt_with_Aii<T, Tvec>(
            cell_sizes,
            block_size,
            cell_global_id,
            graph_iter_xp,
            graph_iter_xm,
            graph_iter_yp,
            graph_iter_ym,
            graph_iter_zp,
            graph_iter_zm,
            field_access,
            Aii_unused);
    }

    /**
     * @brief Get the weight of the Jacobi preconditioner of the 7-point Laplacian
     *
     * Sum over the directions having at least one neighbour of \f$ \Delta x^2 / d \f$, \f$ d \f$
     * being the distance between the cell centers used in laplacian_7pt_with_Aii (\f$ 6 \Delta x
     * \f$ on a uniform grid).
     *
     * TODO: (kept from new_amr_plus_grav) unlike the Aii output of laplacian_7pt_with_Aii (used by
     * CGInit for z_0), a direction without neighbour does not contribute here, so the two differ
     * on the cells missing a neighbour (outer ghost cells, non periodic boundaries).
     *
     * @tparam T field type
     * @tparam Tvec vector type
     * @param cell_sizes cell size per block
     * @param block_size number of cells per block
     * @param cell_global_id id of the cell
     * @param graph_iter_xp neighbour graph accessor in the +x direction
     * @param graph_iter_xm neighbour graph accessor in the -x direction
     * @param graph_iter_yp neighbour graph accessor in the +y direction
     * @param graph_iter_ym neighbour graph accessor in the -y direction
     * @param graph_iter_zp neighbour graph accessor in the +z direction
     * @param graph_iter_zm neighbour graph accessor in the -z direction
     * @return T the Jacobi weight of the cell
     */
    template<class T, class Tvec>
    inline T jacobi_weight_7pt(
        const shambase::VecComponent<Tvec> *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm) {

        auto cur_cell_block_id = cell_global_id / block_size;

        auto get_face_weight = [&](auto &graph_links) -> T {
            auto cell_center_dist = cell_sizes[cur_cell_block_id];

            auto fac = 1.;
            u32 cnt  = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;

                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = 3. / 2.;
                }
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = 3. / 4.;
                }
            });

            return (cnt > 0) ? 1.0 / (cell_center_dist * fac)
                             : shambase::VectorProperties<T>::get_zero();
        };

        T res = shambase::VectorProperties<T>::get_zero();
        res += get_face_weight(graph_iter_xp);
        res += get_face_weight(graph_iter_xm);
        res += get_face_weight(graph_iter_yp);
        res += get_face_weight(graph_iter_ym);
        res += get_face_weight(graph_iter_zp);
        res += get_face_weight(graph_iter_zm);

        auto S = cell_sizes[cur_cell_block_id] * cell_sizes[cur_cell_block_id];
        return S * res;
    }
} // namespace shammodels::basegodunov
