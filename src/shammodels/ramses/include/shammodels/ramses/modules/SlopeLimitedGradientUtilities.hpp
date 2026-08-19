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
 * @file SlopeLimitedGradientUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/slopeLimiter.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include <type_traits>

namespace {
    using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
    using Direction            = shammodels::basegodunov::modules::Direction;

    template<class T>
    inline T slope_function_van_leer_f_form(T sL, T sR) {
        T st = sL + sR;

        auto vanleer = [](T f) {
            return 4. * f * (1. - f);
        };

        auto slopelim = [&](T f) {
            if constexpr (std::is_same_v<T, f64_3>) {
                f.x() = (f.x() >= 0 && f.x() <= 1) ? f.x() : 0;
                f.y() = (f.y() >= 0 && f.y() <= 1) ? f.y() : 0;
                f.z() = (f.z() >= 0 && f.z() <= 1) ? f.z() : 0;
            } else {
                f = (f >= 0 && f <= 1) ? f : 0;
            }
            return vanleer(f);
        };

        return slopelim(sL / st) * st * 0.5;
    }

    template<class T>
    inline T slope_function_van_leer_symetric(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::van_leer_slope_symetric(sL[0], sR[0]),
                shammath::van_leer_slope_symetric(sL[1], sR[1]),
                shammath::van_leer_slope_symetric(sL[2], sR[2])};
        } else {
            return shammath::van_leer_slope_symetric(sL, sR);
        }
    }

    template<class T>
    inline T slope_function_van_leer_standard(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::van_leer_slope(sL[0], sR[0]),
                shammath::van_leer_slope(sL[1], sR[1]),
                shammath::van_leer_slope(sL[2], sR[2])};
        } else {
            return shammath::van_leer_slope(sL, sR);
        }
    }

    template<class T>
    inline T slope_function_minmod(T sL, T sR) {

        if constexpr (std::is_same_v<T, f64_3>) {
            return {
                shammath::minmod(sL[0], sR[0]),
                shammath::minmod(sL[1], sR[1]),
                shammath::minmod(sL[2], sR[2])};
        } else {
            return shammath::minmod(sL, sR);
        }
    }

    using SlopeMode = shammodels::basegodunov::SlopeMode;

    template<class T, SlopeMode mode>
    inline T slope_function(T sL, T sR) {
        if constexpr (mode == SlopeMode::None) {
            return sham::VectorProperties<T>::get_zero();
        }

        if constexpr (mode == SlopeMode::VanLeer_f) {
            return slope_function_van_leer_f_form(sL, sR);
        }

        if constexpr (mode == SlopeMode::VanLeer_std) {
            return slope_function_van_leer_standard(sL, sR);
        }

        if constexpr (mode == SlopeMode::VanLeer_sym) {
            return slope_function_van_leer_symetric(sL, sR);
        }

        if constexpr (mode == SlopeMode::Minmod) {
            return slope_function_minmod(sL, sR);
        }
    }

    /**
     * @brief Get the 3d, slope limited gradient of a field
     *
     * @tparam T
     * @tparam Tvec
     * @tparam mode
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
     * @return std::array<T, 3>
     */
    template<class Tfield, class Tvec, SlopeMode mode, class ACCField>
    inline std::array<Tfield, 3> get_3d_grad(
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

        auto cur_cell_block_id = cell_global_id / block_size;

        auto get_gradient_dir = [&](auto &graph_links, Direction dir) -> Tfield {
            Tfield acc            = shambase::VectorProperties<Tfield>::get_zero();
            auto cell_center_dist = cell_sizes[cur_cell_block_id];

            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;
                auto fac            = 1.;
                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 2.);
                }
                // This logic suppose that the last (4-th) cell at interface have same size with the
                // other three cells. This is also consitent with 2:1 refinement.
                // TODO: extended to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
                const auto inv_dist = 1. / (fac * cell_center_dist);

                int sign = 1 - 2 * (dir % 2);
                acc += sign * inv_dist * (field_access(id_b) - field_access(cell_global_id));
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<Tfield>::get_zero();
        };

        return {
            slope_function<Tfield, mode>(
                get_gradient_dir(graph_iter_xm, Direction::xm),
                get_gradient_dir(graph_iter_xp, Direction::xp)),
            slope_function<Tfield, mode>(
                get_gradient_dir(graph_iter_ym, Direction::ym),
                get_gradient_dir(graph_iter_yp, Direction::yp)),
            slope_function<Tfield, mode>(
                get_gradient_dir(graph_iter_zm, Direction::zm),
                get_gradient_dir(graph_iter_zp, Direction::zp))};
    }

    /**
     * @brief Get the 3d, slope limited gradient of all conservative state
     *
     *
     * @tparam Tvec
     * @tparam mode
     * @tparam ACCField1
     * @tparam ACCField2
     * @tparam ACCField3
     * @param cell_sizes
     * @param block_size
     * @param cell_global_id
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access_rho
     * @param field_access_rhoe
     * @param field_access_rho_vel
     * @return std::array<shammath::ConsState<Tvec>, 3>
     */
    template<class Tvec, SlopeMode mode, class ACCField1, class ACCField2, class ACCField3>
    inline std::array<shammath::ConsState<Tvec>, 3> get_3d_grad_cons(
        const f64 *cell_sizes,
        const u32 block_size,
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField1 &&field_access_rho,
        ACCField2 &&field_access_rho_vel,
        ACCField3 &&field_access_rhoe) {

        using Tscal            = shambase::VecComponent<Tvec>;
        auto cur_cell_block_id = cell_global_id / block_size;

        auto get_gradient_dir = [&](auto &graph_links, Direction dir) -> shammath::ConsState<Tvec> {
            Tscal acc_rho         = shambase::VectorProperties<Tscal>::get_zero();
            Tscal acc_rhoe        = shambase::VectorProperties<Tscal>::get_zero();
            Tvec acc_rho_vel      = shambase::VectorProperties<Tvec>::get_zero();
            auto cell_center_dist = cell_sizes[cur_cell_block_id];

            auto cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                auto neigh_block_id = id_b / block_size;
                auto fac            = 1.;
                if (cell_sizes[neigh_block_id] > cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 2.);
                }
                // This logic suppose that the last (4-th) cell at interface have same size with the
                // other three cells. This is also consitent with 2:1 refinement.
                // TODO: extended to anisotropic mesh
                if (cell_sizes[neigh_block_id] < cell_sizes[cur_cell_block_id]) {
                    fac = (3. / 4.);
                }
                const auto inv_dist = 1. / (fac * cell_center_dist);

                int sign = 1 - 2 * (dir % 2);
                acc_rho += sign * inv_dist
                           * (field_access_rho(id_b) - field_access_rho(cell_global_id));
                acc_rhoe += sign * inv_dist
                            * (field_access_rhoe(id_b) - field_access_rhoe(cell_global_id));
                acc_rho_vel
                    += sign * inv_dist
                       * (field_access_rho_vel(id_b) - field_access_rho_vel(cell_global_id));
            });

            shammath::ConsState<Tvec> res
                = {shambase::VectorProperties<Tscal>::get_zero(),
                   shambase::VectorProperties<Tscal>::get_zero(),

                   {shambase::VectorProperties<Tscal>::get_zero(),
                    shambase::VectorProperties<Tscal>::get_zero(),
                    shambase::VectorProperties<Tscal>::get_zero()}};
            if (cnt > 0) {
                res = {acc_rho, acc_rhoe, acc_rho_vel};
                res *= 1. / cnt;
            }
            return res;
        };

        auto get_avg_neigh = [&](auto &graph_links) -> shammath::ConsState<Tvec> {
            Tscal acc_rho    = shambase::VectorProperties<Tscal>::get_zero();
            Tscal acc_rhoe   = shambase::VectorProperties<Tscal>::get_zero();
            Tvec acc_rho_vel = shambase::VectorProperties<Tvec>::get_zero();
            u32 cnt          = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc_rho += field_access_rho(id_b);
                acc_rho_vel += field_access_rho_vel(id_b);
                acc_rhoe += field_access_rhoe(id_b);
            });

            shammath::ConsState<Tvec> res
                = {shambase::VectorProperties<Tscal>::get_zero(),
                   shambase::VectorProperties<Tscal>::get_zero(),

                   {shambase::VectorProperties<Tscal>::get_zero(),
                    shambase::VectorProperties<Tscal>::get_zero(),
                    shambase::VectorProperties<Tscal>::get_zero()}};

            if (cnt > 0) {
                res = {acc_rho, acc_rhoe, acc_rho_vel};
                res *= (1. / cnt);
            }

            return res;
        };

        shammath::ConsState<Tvec> delta_xp = get_gradient_dir(graph_iter_xp, Direction::xp);
        shammath::ConsState<Tvec> delta_xm = get_gradient_dir(graph_iter_xm, Direction::xm);
        shammath::ConsState<Tvec> delta_yp = get_gradient_dir(graph_iter_yp, Direction::yp);
        shammath::ConsState<Tvec> delta_ym = get_gradient_dir(graph_iter_ym, Direction::ym);
        shammath::ConsState<Tvec> delta_zp = get_gradient_dir(graph_iter_zp, Direction::zp);
        shammath::ConsState<Tvec> delta_zm = get_gradient_dir(graph_iter_zm, Direction::zm);

        shammath::ConsState<Tvec> lim_slope_x
            = {slope_function<Tscal, mode>(delta_xm.rho, delta_xp.rho),
               slope_function<Tscal, mode>(delta_xm.rhoe, delta_xp.rhoe),
               slope_function<Tvec, mode>(delta_xm.rhovel, delta_xp.rhovel)};

        shammath::ConsState<Tvec> lim_slope_y
            = {slope_function<Tscal, mode>(delta_ym.rho, delta_yp.rho),
               slope_function<Tscal, mode>(delta_ym.rhoe, delta_yp.rhoe),
               slope_function<Tvec, mode>(delta_ym.rhovel, delta_yp.rhovel)};

        shammath::ConsState<Tvec> lim_slope_z
            = {slope_function<Tscal, mode>(delta_zm.rho, delta_zp.rho),
               slope_function<Tscal, mode>(delta_zm.rhoe, delta_zp.rhoe),
               slope_function<Tvec, mode>(delta_zm.rhovel, delta_zp.rhovel)};

        return {lim_slope_x, lim_slope_y, lim_slope_z};
    }

    /**
     * @brief Pseudo-gradient
     */
    template<class T, class Tvec, class ACCField>
    inline T get_pseudo_grad(
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access)

    {

        using namespace sham;
        using namespace sham::details;

        auto get_avg_neigh = [&](auto &graph_links, u32 dir) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });

            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        auto epsilon = shambase::get_epsilon<T>();
        T u_cur      = field_access(cell_global_id);
        T u_xp       = get_avg_neigh(graph_iter_xp, 0);
        T u_xm       = get_avg_neigh(graph_iter_xm, 1);
        T u_yp       = get_avg_neigh(graph_iter_yp, 2);
        T u_ym       = get_avg_neigh(graph_iter_ym, 3);
        T u_zp       = get_avg_neigh(graph_iter_zp, 4);
        T u_zm       = get_avg_neigh(graph_iter_zm, 5);

        // RAMSES LIKE

        T x_scal = 2
                   * g_sycl_max(
                       g_sycl_abs((u_cur - u_xm) / (epsilon + u_cur + u_xm)),
                       g_sycl_abs((u_cur - u_xp) / (epsilon + u_cur + u_xp)));

        T y_scal = 2
                   * g_sycl_max(
                       g_sycl_abs((u_cur - u_ym) / (epsilon + u_cur + u_ym)),
                       g_sycl_abs((u_cur - u_yp) / (epsilon + u_cur + u_yp)));
        T z_scal = 2
                   * g_sycl_max(
                       g_sycl_abs((u_cur - u_zm) / (epsilon + u_cur + u_zm)),
                       g_sycl_abs((u_cur - u_zp) / (epsilon + u_cur + u_zp)));

        T res = g_sycl_max(x_scal, g_sycl_max(y_scal, z_scal));
        return res;
    }

    /**
     */
    template<class T, class ACCField>
    inline T baryonic_normalized_slope_criterion(
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access)

    {

        using namespace sham;
        using namespace sham::details;

        auto get_avg_neigh = [&](auto &graph_links, u32 dir) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });

            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        auto epsilon = shambase::get_epsilon<T>();
        T u_cur      = field_access(cell_global_id);
        T u_xp       = get_avg_neigh(graph_iter_xp, 0);
        T u_xm       = get_avg_neigh(graph_iter_xm, 1);
        T u_yp       = get_avg_neigh(graph_iter_yp, 2);
        T u_ym       = get_avg_neigh(graph_iter_ym, 3);
        T u_zp       = get_avg_neigh(graph_iter_zp, 4);
        T u_zm       = get_avg_neigh(graph_iter_zm, 5);

        T norm_slope_x = g_sycl_abs((u_xm - u_xp) / (2 * u_cur + epsilon));
        T norm_slope_y = g_sycl_abs((u_ym - u_yp) / (2 * u_cur + epsilon));
        T norm_slope_z = g_sycl_abs((u_zm - u_zp) / (2 * u_cur + epsilon));

        T res = g_sycl_max(norm_slope_x, g_sycl_max(norm_slope_y, norm_slope_z));

        return res;
    }

    /***
     *  Lohner second order criterion
     */
    template<class T, class Tvec, class ACCField>
    inline T modif_second_derivative(
        const u32 cell_global_id,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {
        using namespace sham;
        using namespace sham::details;

        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        auto eps_ref = 0.01;
        auto epsilon = shambase::get_epsilon<T>();
        T u_cur      = field_access(cell_global_id);
        T u_xp       = get_avg_neigh(graph_iter_xp);
        T u_xm       = get_avg_neigh(graph_iter_xm);
        T u_yp       = get_avg_neigh(graph_iter_yp);
        T u_ym       = get_avg_neigh(graph_iter_ym);
        T u_zp       = get_avg_neigh(graph_iter_zp);
        T u_zm       = get_avg_neigh(graph_iter_zm);

        T delta_u_xp = u_xp - u_cur;
        T delta_u_xm = u_xm - u_cur;
        T delta_u_yp = u_yp - u_cur;
        T delta_u_ym = u_ym - u_cur;
        T delta_u_zp = u_zp - u_cur;
        T delta_u_zm = u_zm - u_cur;

        T scalar_x = g_sycl_abs(u_xp) + g_sycl_abs(u_xm) + 2 * g_sycl_abs(u_cur);
        T scalar_y = g_sycl_abs(u_yp) + g_sycl_abs(u_ym) + 2 * g_sycl_abs(u_cur);
        T scalar_z = g_sycl_abs(u_zp) + g_sycl_abs(u_zm) + 2 * g_sycl_abs(u_cur);

        T res_x
            = g_sycl_abs(delta_u_xm + delta_u_xp)
              / (g_sycl_abs(delta_u_xm) + g_sycl_abs(delta_u_xp) + eps_ref * scalar_x + epsilon);
        T res_y
            = g_sycl_abs(delta_u_ym + delta_u_yp)
              / (g_sycl_abs(delta_u_ym) + g_sycl_abs(delta_u_yp) + eps_ref * scalar_y + epsilon);
        T res_z
            = g_sycl_abs(delta_u_zm + delta_u_zp)
              / (g_sycl_abs(delta_u_zm) + g_sycl_abs(delta_u_zp) + eps_ref * scalar_z + epsilon);

        return (res_x + res_y + res_z);
    }

    /**
     * @brief Normalized Shear criterion
     */
    template<class Tvec, class ACCField>
    inline shambase::VecComponent<Tvec> normalized_shear(
        const u32 cell_global_id,
        const f32 sound_speed,
        const Tvec delta_cells,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access)

    {

        using namespace sham;
        using namespace sham::details;

        auto get_avg_neigh = [&](auto &graph_links, u32 dir) -> Tvec {
            Tvec acc = shambase::VectorProperties<Tvec>::get_zero();
            u32 cnt  = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });

            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<Tvec>::get_zero();
        };

        Tvec u_xp = get_avg_neigh(graph_iter_xp, 0);
        Tvec u_xm = get_avg_neigh(graph_iter_xm, 1);
        Tvec u_yp = get_avg_neigh(graph_iter_yp, 2);
        Tvec u_ym = get_avg_neigh(graph_iter_ym, 3);
        Tvec u_zp = get_avg_neigh(graph_iter_zp, 4);
        Tvec u_zm = get_avg_neigh(graph_iter_zm, 5);

        auto vgy = 0.25 * (u_xp[1] - u_xm[1]) * (u_xp[1] - u_xm[1]);
        auto vgx = 0.25 * (u_yp[0] - u_ym[0]) * (u_yp[0] - u_ym[0]);

        return vgy + vgx;

        /**---------------
        * Shear criterion in 3D
        ------------------*/
        // auto dv_xdir = 0.5 * sycl::abs(u_xp - u_xm);
        // auto dv_ydir = 0.5 * sycl::abs(u_yp - u_ym);
        // auto dv_zdir = 0.5 * sycl::abs(u_zp - u_zm) ;
        // auto shear_1 =  (dv_ydir[0] + dv_xdir[1])*(dv_ydir[0] + dv_xdir[1]);
        // auto shear_2 = (dv_ydir[2] + dv_zdir[1]) * (dv_ydir[2] + dv_zdir[1]);
        // auto shear_3 = (dv_zdir[0] + dv_xdir[2]) * (dv_zdir[0] + dv_xdir[2]);
        // return  (shear_1 + shear_2 + shear_3) * (delta_cells.x() * delta_cells.x())/(sound_speed
        // * sound_speed);
    }

} // namespace
