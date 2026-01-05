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
 * @file SlopeLimitedGradientUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/slopeLimiter.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include <type_traits>
#include <cmath>

using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

namespace {

    using Direction = shammodels::basegodunov::modules::Direction;

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
        const u32 cell_global_id,
        const shambase::VecComponent<Tvec> delta_cell,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField &&field_access) {

        auto get_avg_neigh = [&](auto &graph_links) -> Tfield {
            Tfield acc = shambase::VectorProperties<Tfield>::get_zero();
            u32 cnt    = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            return (cnt > 0) ? acc / cnt : shambase::VectorProperties<Tfield>::get_zero();
        };

        Tfield W_i  = field_access(cell_global_id);
        Tfield W_xp = get_avg_neigh(graph_iter_xp);
        Tfield W_xm = get_avg_neigh(graph_iter_xm);
        Tfield W_yp = get_avg_neigh(graph_iter_yp);
        Tfield W_ym = get_avg_neigh(graph_iter_ym);
        Tfield W_zp = get_avg_neigh(graph_iter_zp);
        Tfield W_zm = get_avg_neigh(graph_iter_zm);

        Tfield delta_W_x_p = W_xp - W_i;
        Tfield delta_W_y_p = W_yp - W_i;
        Tfield delta_W_z_p = W_zp - W_i;

        Tfield delta_W_x_m = W_i - W_xm;
        Tfield delta_W_y_m = W_i - W_ym;
        Tfield delta_W_z_m = W_i - W_zm;

        Tfield fact = 1. / Tfield(delta_cell);

        Tfield lim_slope_W_x = slope_function<Tfield, mode>(delta_W_x_m * fact, delta_W_x_p * fact);
        Tfield lim_slope_W_y = slope_function<Tfield, mode>(delta_W_y_m * fact, delta_W_y_p * fact);
        Tfield lim_slope_W_z = slope_function<Tfield, mode>(delta_W_z_m * fact, delta_W_z_p * fact);

        return {lim_slope_W_x, lim_slope_W_y, lim_slope_W_z};
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
     * @param cell_global_id
     * @param delta_cell
     * @param graph_iter_xp
     * @param graph_iter_xm
     * @param graph_iter_yp
     * @param graph_iter_ym
     * @param graph_iter_zp
     * @param graph_iter_zm
     * @param field_access
     * @return std::array<shammath::ConsState<Tvec>, 3>
     */
    template<class Tvec, SlopeMode mode, class ACCField1, class ACCField2, class ACCField3>
    inline std::array<shammath::ConsState<Tvec>, 3> get_3d_grad_cons(
        const u32 cell_global_id,
        const shambase::VecComponent<Tvec> delta_cell,
        const AMRGraphLinkiterator &graph_iter_xp,
        const AMRGraphLinkiterator &graph_iter_xm,
        const AMRGraphLinkiterator &graph_iter_yp,
        const AMRGraphLinkiterator &graph_iter_ym,
        const AMRGraphLinkiterator &graph_iter_zp,
        const AMRGraphLinkiterator &graph_iter_zm,
        ACCField1 &&field_access_rho,
        ACCField2 &&field_access_rho_vel,
        ACCField3 &&field_access_rhoe) {

        using Tscal = shambase::VecComponent<Tvec>;

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
                res *= (1. / cnt);
            }

            return res;
        };

        shammath::ConsState<Tvec> W_i
            = {field_access_rho(cell_global_id),
               field_access_rhoe(cell_global_id),
               field_access_rho_vel(cell_global_id)};

        // logger::raw_ln("id_grad ", cell_global_id ,"global_id rho  grad  ",
        // field_access_rho(cell_global_id));

        shammath::ConsState<Tvec> W_xp = get_avg_neigh(graph_iter_xp);
        shammath::ConsState<Tvec> W_xm = get_avg_neigh(graph_iter_xm);
        shammath::ConsState<Tvec> W_yp = get_avg_neigh(graph_iter_yp);
        shammath::ConsState<Tvec> W_ym = get_avg_neigh(graph_iter_ym);
        shammath::ConsState<Tvec> W_zp = get_avg_neigh(graph_iter_zp);
        shammath::ConsState<Tvec> W_zm = get_avg_neigh(graph_iter_zm);

        shammath::ConsState<Tvec> delta_W_x_p = W_xp - W_i;
        shammath::ConsState<Tvec> delta_W_y_p = W_yp - W_i;
        shammath::ConsState<Tvec> delta_W_z_p = W_zp - W_i;

        shammath::ConsState<Tvec> delta_W_x_m = W_i - W_xm;
        shammath::ConsState<Tvec> delta_W_y_m = W_i - W_ym;
        shammath::ConsState<Tvec> delta_W_z_m = W_i - W_zm;

        Tscal fact = 1. / delta_cell;

        shammath::ConsState<Tvec> lim_slope_W_x
            = {slope_function<Tscal, mode>(delta_W_x_m.rho * fact, delta_W_x_p.rho * fact),
               slope_function<Tscal, mode>(delta_W_x_m.rhoe * fact, delta_W_x_p.rhoe * fact),
               slope_function<Tvec, mode>(delta_W_x_m.rhovel * fact, delta_W_x_p.rhovel * fact)};

        shammath::ConsState<Tvec> lim_slope_W_y
            = {slope_function<Tscal, mode>(delta_W_y_m.rho * fact, delta_W_y_p.rho * fact),
               slope_function<Tscal, mode>(delta_W_y_m.rhoe * fact, delta_W_y_p.rhoe * fact),
               slope_function<Tvec, mode>(delta_W_y_m.rhovel * fact, delta_W_y_p.rhovel * fact)};

        shammath::ConsState<Tvec> lim_slope_W_z
            = {slope_function<Tscal, mode>(delta_W_z_m.rho * fact, delta_W_z_p.rho * fact),
               slope_function<Tscal, mode>(delta_W_z_m.rhoe * fact, delta_W_z_p.rhoe * fact),
               slope_function<Tvec, mode>(delta_W_z_m.rhovel * fact, delta_W_z_p.rhovel * fact)};

        return {lim_slope_W_x, lim_slope_W_y, lim_slope_W_z};
    }

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
/*
        auto grad_scalar = [&](T u_curr, T u_neigh) {
            T max = g_sycl_max(g_sycl_abs(u_curr), g_sycl_abs(u_neigh));
            max   = g_sycl_abs(u_curr - u_neigh) / max;
            return g_sycl_max(g_sycl_min(max, 1.0), 0.0);
        };

        auto get_amr_grad = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                T u_cur_acc   = field_access(cell_global_id);
                T u_neigh_acc = field_access(id_b);
                acc           = g_sycl_max(acc, grad_scalar(u_cur_acc, u_neigh_acc));
            });

            return (cnt > 0) ? acc : shambase::VectorProperties<T>::get_zero();
        };

        T u_xp_dir = get_amr_grad(graph_iter_xp);
        T u_xm_dir = get_amr_grad(graph_iter_xm);
        T u_yp_dir = get_amr_grad(graph_iter_yp);
        T u_ym_dir = get_amr_grad(graph_iter_ym);
        T u_zp_dir = get_amr_grad(graph_iter_zp);
        T u_zm_dir = get_amr_grad(graph_iter_zm);

        T res = max_8points(
            shambase::VectorProperties<T>::get_zero(),
            shambase::VectorProperties<T>::get_zero(),
            u_xp_dir,
            u_xm_dir,
            u_yp_dir,
            u_ym_dir,
            u_zp_dir,
            u_zm_dir);
        */
        auto get_avg_neigh = [&](auto &graph_links) -> T {
            T acc   = shambase::VectorProperties<T>::get_zero();
            u32 cnt = graph_links.for_each_object_link_cnt(cell_global_id, [&](u32 id_b) {
                acc += field_access(id_b);
            });
            // if(cnt > 1)
                // logger::info_ln("cell [",cell_global_id,"] : ", cnt, "\t\n");
            if (cnt == 1 || (cnt >1  && cnt != 4)){
                return acc / cnt;
            }
            else return  0;
            // return (cnt > 0) ? acc / cnt : shambase::VectorProperties<T>::get_zero();
        };

        auto epsilon = shambase::get_epsilon<T>();
        T u_cur      = field_access(cell_global_id);
        // T u_xp = 0;
        // T u_xm = 0;
        T u_xp       = get_avg_neigh(graph_iter_xp);
        T u_xm       = get_avg_neigh(graph_iter_xm);
        T u_yp       = get_avg_neigh(graph_iter_yp);
        T u_ym       = get_avg_neigh(graph_iter_ym);
        T u_zp       = get_avg_neigh(graph_iter_zp);
        T u_zm       = get_avg_neigh(graph_iter_zm);

        T x_scal =  2 * g_sycl_max(g_sycl_abs((u_cur - u_xm)/ (epsilon + u_cur + u_xm)), g_sycl_abs((u_cur - u_xp)/ (epsilon + u_cur + u_xp)));
        T y_scal = 0.;
        T z_scal = 0.; 
        // T y_scal =  2 * g_sycl_max(g_sycl_abs((u_cur - u_ym)/ (epsilon + u_cur + u_ym)), g_sycl_abs((u_cur - u_yp)/ (epsilon + u_cur + u_yp)));
        // T z_scal =  2 * g_sycl_max(g_sycl_abs((u_cur - u_zm)/ (epsilon + u_cur + u_zm)), g_sycl_abs((u_cur - u_zp)/ (epsilon + u_cur + u_zp)));

        T res = g_sycl_max(x_scal, g_sycl_max(y_scal, z_scal));

        return res;
    }

    template<class T, class Tvec, class ACCField>
    inline T modif_second_derivative(
        const u32 cell_global_id,
        // const shambase::VecComponent<Tvec> delta_cell,
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

        // return g_sycl_max(res_x, g_sycl_max(res_y, res_z)) * delta_cell;
        // return g_sycl_max(res_x, g_sycl_max(res_y, res_z));
        return (res_x + res_y + res_z);
    }

    template<class T, class Tvec, class ACCField>
    inline T get_Lohner(
        const u32 cell_global_id,
        // const shambase::VecComponent<Tvec> delta_cell,
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

        T d_uxx = u_xm - 2 * u_cur + u_xp;
        T d_uyy = u_ym - 2 * u_cur + u_yp;
        T d_uzz = u_zm - 2 * u_cur + u_zp;

        T num_  = d_uxx * d_uxx + d_uyy * d_uyy + d_uzz * d_uzz;
        T den_x = g_sycl_abs(delta_u_xm) + g_sycl_abs(delta_u_xp);
        T den_y = g_sycl_abs(delta_u_ym) + g_sycl_abs(delta_u_yp);
        T den_z = g_sycl_abs(delta_u_zm) + g_sycl_abs(delta_u_zp);
        T eps_  = eps_ref * g_sycl_abs(u_cur);

        // T E = (num_) / (den_x * den_x + den_y * den_y + den_z * den_z + eps_ * eps_);
        // E = sycl::sqrt(E);

        T E = (d_uxx / (den_x + eps_)) * (d_uxx / (den_x + eps_))
              + (d_uyy / (den_y + eps_)) * (d_uyy / (den_y + eps_))
              + (d_uzz / (den_z + eps_)) * (d_uzz / (den_z + eps_));
        // E = sycl::sqrt(E);

        return E;
    }

} // namespace
