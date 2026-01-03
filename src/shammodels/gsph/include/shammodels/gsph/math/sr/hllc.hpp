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
 * @file hllc.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief SR Riemann Solver interface for Special Relativistic GSPH
 *
 * This file provides backward-compatible wrappers that call the exact
 * Riemann solver from exact.hpp. The exact solver uses Newton-Raphson
 * iteration following Pons et al. (2000) and Kitajima et al. (2025).
 *
 * The HLLC naming is kept for API compatibility but the implementation
 * now uses the exact solver for improved accuracy.
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/gsph/math/sr/exact.hpp"

namespace shammodels::gsph::sr {

    /**
     * @brief Left/Right state for SR Riemann problem
     */
    template<class Tscal>
    struct SRRiemannState {
        Tscal v_x;         ///< Normal velocity component
        Tscal v_t;         ///< Tangent velocity component
        Tscal density;     ///< Rest-frame density n
        Tscal pressure;    ///< Pressure P
        Tscal sound_speed; ///< Sound speed c_s (unused, kept for API compatibility)
        Tscal enthalpy;    ///< Specific enthalpy H (unused, kept for API compatibility)
    };

    /**
     * @brief Result of SR Riemann solver
     */
    template<class Tscal>
    struct SRRiemannResult {
        Tscal P_star;   ///< Interface pressure
        Tscal v_x_star; ///< Interface normal velocity
        Tscal v_t_star; ///< Interface tangent velocity
    };

    /**
     * @brief Compute relativistic characteristic speeds
     *
     * For relativistic flows, the characteristic speeds are:
     *   λ± = (v ± c_s) / (1 ± v*c_s/c²)
     *
     * @param v_x Normal velocity
     * @param c_s Sound speed
     * @param c Speed of light
     * @param is_plus True for λ+, false for λ-
     * @return Characteristic speed
     */
    template<class Tscal>
    inline Tscal relativistic_characteristic_speed(Tscal v_x, Tscal c_s, Tscal c, bool is_plus) {

        const Tscal c2 = c * c;
        if (is_plus) {
            const Tscal denom = Tscal{1} + v_x * c_s / c2;
            return (denom > Tscal{1e-10}) ? (v_x + c_s) / denom : c;
        } else {
            const Tscal denom = Tscal{1} - v_x * c_s / c2;
            return (denom > Tscal{1e-10}) ? (v_x - c_s) / denom : -c;
        }
    }

    /**
     * @brief SR Riemann solver (calls exact solver internally)
     *
     * This is a wrapper for backward compatibility. The actual implementation
     * uses the exact Newton-Raphson solver from exact.hpp.
     *
     * @param left Left state
     * @param right Right state
     * @param gamma_eos Adiabatic index γ_c
     * @param c_speed Speed of light (unused, c=1 assumed)
     * @return Interface state (P*, v_x*, v_t*)
     */
    template<class Tscal>
    inline SRRiemannResult<Tscal> sr_hllc_solver(
        const SRRiemannState<Tscal> &left,
        const SRRiemannState<Tscal> &right,
        Tscal gamma_eos,
        Tscal c_speed) {

        SRRiemannResult<Tscal> result;

        // Call exact solver
        ExactRiemannResult<Tscal> exact_result = sr_exact_solver(
            left.v_x,
            left.v_t,
            left.density,
            left.pressure,
            right.v_x,
            right.v_t,
            right.density,
            right.pressure,
            gamma_eos);

        result.P_star   = exact_result.P_star;
        result.v_x_star = exact_result.v_x_star;
        result.v_t_star = exact_result.v_t_star;

        return result;
    }

    /**
     * @brief Build Riemann state from primitive variables
     *
     * @param v_x Normal velocity
     * @param v_t Tangent velocity
     * @param density Rest-frame density
     * @param pressure Pressure
     * @param gamma_eos Adiabatic index
     * @param c_speed Speed of light
     * @return SRRiemannState structure
     */
    template<class Tscal>
    inline SRRiemannState<Tscal> build_riemann_state(
        Tscal v_x, Tscal v_t, Tscal density, Tscal pressure, Tscal gamma_eos, Tscal c_speed) {

        SRRiemannState<Tscal> state;
        state.v_x      = v_x;
        state.v_t      = v_t;
        state.density  = sycl::fmax(density, Tscal{1e-10});
        state.pressure = sycl::fmax(pressure, Tscal{1e-6});

        // Enthalpy: H = 1 + (γ/(γ-1)) * P/n
        state.enthalpy
            = Tscal{1} + (gamma_eos / (gamma_eos - Tscal{1})) * state.pressure / state.density;

        // Sound speed: c_s² = (γ-1)(H-1)/H * c²
        const Tscal cs2 = (gamma_eos - Tscal{1}) * (state.enthalpy - Tscal{1}) / state.enthalpy
                          * c_speed * c_speed;
        state.sound_speed = sycl::sqrt(sycl::fmax(cs2, Tscal{0}));

        return state;
    }

    /**
     * @brief SR Riemann solver interface for pairwise interactions
     *
     * This is the main entry point for SR-GSPH force computation.
     * Takes primitive variables and returns interface state.
     * Internally calls the exact Newton-Raphson solver.
     *
     * @param v_x_L Left normal velocity
     * @param v_t_L Left tangent velocity
     * @param rho_L Left rest-frame density n
     * @param P_L Left pressure
     * @param v_x_R Right normal velocity
     * @param v_t_R Right tangent velocity
     * @param rho_R Right rest-frame density n
     * @param P_R Right pressure
     * @param gamma_eos Adiabatic index
     * @param c_speed Speed of light (unused, c=1 assumed)
     * @param P_star Output: interface pressure
     * @param v_x_star Output: interface normal velocity
     * @param v_t_star Output: interface tangent velocity
     */
    template<class Tscal>
    inline void sr_hllc_interface(
        Tscal v_x_L,
        Tscal v_t_L,
        Tscal rho_L,
        Tscal P_L,
        Tscal v_x_R,
        Tscal v_t_R,
        Tscal rho_R,
        Tscal P_R,
        Tscal gamma_eos,
        Tscal c_speed,
        Tscal &P_star,
        Tscal &v_x_star,
        Tscal &v_t_star) {

        // Call exact solver directly
        sr_exact_interface(
            v_x_L,
            v_t_L,
            rho_L,
            P_L,
            v_x_R,
            v_t_R,
            rho_R,
            P_R,
            gamma_eos,
            P_star,
            v_x_star,
            v_t_star);
    }

} // namespace shammodels::gsph::sr
