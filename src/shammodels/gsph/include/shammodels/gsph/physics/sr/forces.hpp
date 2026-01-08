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
 * @file forces.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Force computation for Special Relativistic GSPH
 *
 * Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 *
 * Key formulas (Kitajima Eq. 371-374):
 *   <nu_i * dS_i/dt> = -Sum_j nu_j P*_ij V^2_ij,interp [nabla_i W - nabla_j W]
 *   <nu_i * de_i/dt> = -Sum_j nu_j P*_ij v*_ij . V^2_ij,interp [nabla_i W - nabla_j W]
 *
 * For the volume-based approach (Kitajima Eq. 221, 284):
 *   V_p = 1/W_sum (particle volume, independent of baryon number)
 *   V^2_ij,interp = (V_i^2 + V_j^2) / 2  (symmetric interpolation)
 *
 * Note: V = nu/N = nu/(nu/V_p) = V_p, so particle volume is independent of nu!
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief Full SR force computation for a particle pair (Kitajima GSPH formulation)
     *
     * Implements Kitajima Eq. 371-374:
     *   <nu_i * dS_i/dt> = -Sum_j nu_j P* V^2_ij,interp [nabla_i W - nabla_j W]
     *   <nu_i * de_i/dt> = -Sum_j nu_j P* v* . V^2_ij,interp [nabla_i W - nabla_j W]
     *
     * where V^2_ij,interp = (V_i^2 + V_j^2) / 2 with V = V_p = h^3/hfact^3
     *
     * NOTE: The returned dS_i and de_i do NOT include nu_j - caller must multiply!
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param P_star Interface pressure from Riemann solver
     * @param v_x_star Interface normal velocity from Riemann solver
     * @param v_t_star Interface tangent velocity from Riemann solver
     * @param n_ij Unit vector from i to j
     * @param v_t_dir_L Left tangent velocity direction (unit vector)
     * @param v_t_dir_R Right tangent velocity direction (unit vector)
     * @param V_i Particle volume of i: V_p = h_i^3 / hfact^3
     * @param V_j Particle volume of j: V_p = h_j^3 / hfact^3
     * @param grad_W_i Kernel gradient for particle i (using sqrt(2)*h_i)
     * @param grad_W_j Kernel gradient for particle j (using sqrt(2)*h_j)
     * @param dS_i Output: momentum derivative contribution (to be divided by nu_i later)
     * @param de_i Output: energy derivative contribution (to be divided by nu_i later)
     */
    template<class Tscal, class Tvec>
    inline void sr_pairwise_force(
        Tscal P_star,
        Tscal v_x_star,
        Tscal v_t_star,
        Tvec n_ij,
        Tvec v_t_dir_L,
        Tvec v_t_dir_R,
        Tscal V_i,
        Tscal V_j,
        Tvec grad_W_i,
        Tvec grad_W_j,
        Tvec &dS_i,
        Tscal &de_i) {

        // Kitajima Eq. 365: V^2_ij = (V^2_ij(h_i) + V^2_ij(h_j)) / 2
        // For volume-based approach: V = V_p = h^3/hfact^3 (independent of nu!)
        // Symmetric interpolation: V^2_interp = (V_i^2 + V_j^2) / 2
        const Tscal V2_interp = (V_i * V_i + V_j * V_j) * Tscal{0.5};

        // Kitajima Eq. 371: <nu_i dS_i/dt> = -Sum_j P* V^2_ij [nabla_i W - nabla_j W]
        // Note: grad_W_i + grad_W_j = nabla_i W - nabla_j W (due to gradient convention)
        Tvec force = (grad_W_i + grad_W_j) * (-P_star * V2_interp);

        // Reconstruct v* vector (use upwind direction for tangent velocity)
        Tvec v_t_dir = (v_x_star > Tscal{0}) ? v_t_dir_L : v_t_dir_R;
        Tvec v_star  = n_ij * v_x_star + v_t_dir * v_t_star;

        // Output: these are <nu_i * dS_i/dt> and <nu_i * de_i/dt>
        // Caller must divide by nu_i to get per-baryon rates
        dS_i = force;
        de_i = sycl::dot(v_star, force);
    }

    /**
     * @brief Compute tangent velocity vector
     *
     * v_t_vec = v - (v . n_ij) * n_ij
     */
    template<class Tscal, class Tvec>
    inline Tvec sr_tangent_velocity_vec(Tvec v, Tvec n_ij) {
        const Tscal v_x = sycl::dot(v, n_ij);
        return v - n_ij * v_x;
    }

} // namespace shammodels::gsph::physics::sr
