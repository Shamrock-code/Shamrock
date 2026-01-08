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
 * @file HLL.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief HLL approximate Riemann solver for Newtonian GSPH
 *
 * Implements the Harten-Lax-van Leer (1983) 2-wave approximate solver.
 * Uses only S_L and S_R wave speeds (no contact wave S*).
 *
 * References:
 * - Harten, Lax, van Leer (1983) "On Upstream Differencing and Godunov-Type
 *   Schemes for Hyperbolic Conservation Laws"
 */

#include "Iterative.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shammodels::gsph::physics::newtonian::riemann {

    /**
     * @brief HLL approximate Riemann solver
     *
     * Harten-Lax-van Leer (1983) 2-wave approximate solver.
     * Uses only S_L and S_R wave speeds (no contact wave S*).
     * Note: This is HLL, not HLLC. HLLC would require the contact wave.
     *
     * @tparam Tscal Scalar type (f32 or f64)
     * @param u_L Left state velocity
     * @param rho_L Left state density
     * @param p_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param p_R Right state pressure
     * @param gamma Adiabatic index
     * @return Result with p_star and v_star
     */
    template<class Tscal>
    inline Result<Tscal> solve_hll(
        Tscal u_L, Tscal rho_L, Tscal p_L, Tscal u_R, Tscal rho_R, Tscal p_R, Tscal gamma) {

        Result<Tscal> result;
        const Tscal smallval = Tscal{1.0e-25};

        // Compute Eulerian sound speeds
        const Tscal c_L = sycl::sqrt(gamma * p_L / sycl::fmax(rho_L, smallval));
        const Tscal c_R = sycl::sqrt(gamma * p_R / sycl::fmax(rho_R, smallval));

        // Roe averages for wave speed estimates
        const Tscal sqrt_rho_L = sycl::sqrt(rho_L);
        const Tscal sqrt_rho_R = sycl::sqrt(rho_R);
        const Tscal roe_inv    = Tscal{1} / (sqrt_rho_L + sqrt_rho_R + smallval);

        const Tscal u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * roe_inv;
        const Tscal c_roe = (sqrt_rho_L * c_L + sqrt_rho_R * c_R) * roe_inv;

        // Wave speed estimates (following reference implementation)
        const Tscal S_L = sycl::fmin(u_L - c_L, u_roe - c_roe);
        const Tscal S_R = sycl::fmax(u_R + c_R, u_roe + c_roe);

        // HLL flux formula (following reference g_fluid_force.cpp hll_solver)
        // c1 = rho_L * (S_L - u_L)
        // c2 = rho_R * (S_R - u_R)
        // c3 = 1 / (c1 - c2)
        // c4 = p_L - u_L * c1
        // c5 = p_R - u_R * c2
        // v* = (c5 - c4) * c3
        // p* = (c1 * c5 - c2 * c4) * c3
        const Tscal c1 = rho_L * (S_L - u_L);
        const Tscal c2 = rho_R * (S_R - u_R);
        const Tscal c3 = Tscal{1} / (c1 - c2 + smallval);
        const Tscal c4 = p_L - u_L * c1;
        const Tscal c5 = p_R - u_R * c2;

        result.v_star = (c5 - c4) * c3;
        result.p_star = sycl::fmax(smallval, (c1 * c5 - c2 * c4) * c3);

        return result;
    }

} // namespace shammodels::gsph::physics::newtonian::riemann
