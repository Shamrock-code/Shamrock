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
     * @param field_access_rho
     * @param field_access_rhoe
     * @param field_access_rho_vel
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
                res = {acc_rho, acc_rhoe, acc_rho_vel};
                res *= (1. / cnt);
            }

            return res;
        };

        shammath::ConsState<Tvec> W_i
            = {field_access_rho(cell_global_id),
               field_access_rhoe(cell_global_id),
               field_access_rho_vel(cell_global_id)};

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
} // namespace
