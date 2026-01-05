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
 * Equations 371-375 for momentum and energy evolution with variable smoothing length.
 *
 * Key formulas (Kitajima Eq. 371-372):
 *   ⟨νᵢṠᵢ⟩ = -Σⱼ P*ᵢⱼ V²ᵢⱼ,interp [∇ᵢW(xᵢ-xⱼ, √2hᵢ) - ∇ⱼW(xᵢ-xⱼ, √2hⱼ)]
 *   ⟨νᵢėᵢ⟩ = -Σⱼ P*ᵢⱼ v*ᵢⱼ · V²ᵢⱼ,interp [∇ᵢW - ∇ⱼW]
 *
 * where (Kitajima Eq. 365):
 *   V²ᵢⱼ,interp = (1/2)(V²ᵢⱼ(hᵢ) + V²ᵢⱼ(hⱼ))  - single interpolated volume
 *   V = ν/N (volume per particle)
 *   [∇ᵢW - ∇ⱼW] = antisymmetric gradient difference
 *
 * Note: This uses the INTERPOLATED V² formulation from the paper,
 * NOT the separate V²Ω grad-h corrected form.
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::sr {

    /**
     * @brief Compute SR force contribution for a particle pair (interpolated V² formula)
     *
     * Kitajima et al. (2025) Eq. 371 with variable smoothing length:
     *   dS_i/dt += -P* * V²ᵢⱼ,interp * (∇W_i + ∇W_j)
     *
     * where V²ᵢⱼ,interp = 0.5 * (V_i² + V_j²) is the interpolated volume factor.
     *
     * Note: The gradient signs are such that ∇W_i and ∇W_j both point in
     * the same direction (from j to i), so their sum gives the antisymmetric
     * form [∇ᵢW - ∇ⱼW] from the paper (since ∇ⱼW points opposite to ∇ᵢW).
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param P_star Interface pressure from Riemann solver
     * @param V_i Volume of particle i (ν_i / N_i)
     * @param V_j Volume of particle j (ν_j / N_j)
     * @param grad_W_i Kernel gradient for particle i (points from j to i)
     * @param grad_W_j Kernel gradient for particle j (points from j to i)
     * @return Force contribution to dS_i/dt
     */
    template<class Tscal, class Tvec>
    inline Tvec sr_momentum_force(
        Tscal P_star,
        Tscal V_i,
        Tscal V_j,
        Tvec grad_W_i,
        Tvec grad_W_j) {

        // Kitajima Eq. 365: Interpolated volume factor
        const Tscal V2_interp = Tscal{0.5} * (V_i * V_i + V_j * V_j);

        // Kitajima Eq. 371: Force with interpolated V²
        // F = -P* * V²_interp * (∇W_i + ∇W_j)
        return (grad_W_i + grad_W_j) * (-P_star * V2_interp);
    }

    /**
     * @brief Compute SR force contribution with grad-h correction
     *
     * Grad-h corrected force formula (Price 2012, Hopkins 2013):
     *   dS_i/dt += -P* * (V_i²/Ω_i * ∇W_i + V_j²/Ω_j * ∇W_j)
     *
     * where Ω = 1 + (h/(3ρ)) * Σⱼ mⱼ ∂W/∂h is the grad-h correction factor.
     * This accounts for spatial variation of h and ensures conservation.
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param P_star Interface pressure from Riemann solver
     * @param V_i Volume of particle i
     * @param V_j Volume of particle j
     * @param omega_i Grad-h correction factor for particle i
     * @param omega_j Grad-h correction factor for particle j
     * @param grad_W_i Kernel gradient for particle i
     * @param grad_W_j Kernel gradient for particle j
     * @return Force contribution to dS_i/dt
     */
    template<class Tscal, class Tvec>
    inline Tvec sr_momentum_force(
        Tscal P_star,
        Tscal V_i,
        Tscal V_j,
        Tscal omega_i,
        Tscal omega_j,
        Tvec grad_W_i,
        Tvec grad_W_j) {

        // Grad-h corrected force: F = -P* * (V_i²/Ω_i * ∇W_i + V_j²/Ω_j * ∇W_j)
        // Using inv_sat_zero to handle omega = 0 case safely
        const Tscal V2_omega_i = V_i * V_i * sham::inv_sat_zero(omega_i);
        const Tscal V2_omega_j = V_j * V_j * sham::inv_sat_zero(omega_j);

        return -(grad_W_i * V2_omega_i + grad_W_j * V2_omega_j) * P_star;
    }

    /**
     * @brief Compute SR energy evolution contribution for a particle pair
     *
     * Implements Kitajima Eq. 59 for energy evolution:
     *   de_i/dt += -P* v* · (V_i² Ω_i ∇W_i + V_j² Ω_j ∇W_j)
     *
     * This is simply v* · force where force is from sr_momentum_force.
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param v_star Interface velocity vector from Riemann solver
     * @param force Force contribution (output of sr_momentum_force)
     * @return Energy evolution contribution to de_i/dt
     */
    template<class Tscal, class Tvec>
    inline Tscal sr_energy_evolution(Tvec v_star, Tvec force) {
        return sycl::dot(v_star, force);
    }

    /**
     * @brief Compute particle volume from SPH number and lab-frame density
     *
     * V_i = ν_i / N_i
     *
     * where ν_i is the SPH particle number (related to smoothing length)
     * and N_i is the lab-frame baryon density.
     *
     * @tparam Tscal Scalar type
     * @param nu SPH particle number (= n_ngb / kernel_norm for adaptive h)
     * @param N Lab-frame density (= γ * n, where n is rest-frame density)
     * @return Particle volume
     */
    template<class Tscal>
    inline Tscal sr_particle_volume(Tscal nu, Tscal N) {
        return nu / sycl::fmax(N, Tscal{1e-15});
    }

    /**
     * @brief Compute lab-frame density from rest-frame density and Lorentz factor
     *
     * N = γ * n
     *
     * @tparam Tscal Scalar type
     * @param n Rest-frame density
     * @param gamma_lor Lorentz factor γ
     * @return Lab-frame density N
     */
    template<class Tscal>
    inline Tscal sr_lab_frame_density(Tscal n, Tscal gamma_lor) {
        return gamma_lor * n;
    }

    /**
     * @brief Reconstruct velocity vector from normal and tangent components
     *
     * For GSPH, we decompose velocity into normal (along r_ij) and tangent
     * components. This function reconstructs the full velocity vector.
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param v_x Normal velocity component
     * @param v_t Tangent velocity magnitude
     * @param n_ij Unit vector from i to j
     * @param v_t_dir Tangent velocity direction (unit vector, perpendicular to n_ij)
     * @return Full velocity vector
     */
    template<class Tscal, class Tvec>
    inline Tvec sr_reconstruct_velocity(Tscal v_x, Tscal v_t, Tvec n_ij, Tvec v_t_dir) {

        return n_ij * v_x + v_t_dir * v_t;
    }

    /**
     * @brief Project velocity onto normal direction
     *
     * v_x = v · n_ij
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param v Velocity vector
     * @param n_ij Unit vector from i to j
     * @return Normal velocity component
     */
    template<class Tscal, class Tvec>
    inline Tscal sr_normal_velocity(Tvec v, Tvec n_ij) {
        return sycl::dot(v, n_ij);
    }

    /**
     * @brief Compute tangent velocity vector
     *
     * v_t_vec = v - (v · n_ij) * n_ij
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param v Velocity vector
     * @param n_ij Unit vector from i to j
     * @return Tangent velocity vector
     */
    template<class Tscal, class Tvec>
    inline Tvec sr_tangent_velocity_vec(Tvec v, Tvec n_ij) {
        const Tscal v_x = sycl::dot(v, n_ij);
        return v - n_ij * v_x;
    }

    /**
     * @brief Compute tangent velocity magnitude
     *
     * |v_t| = |v - (v · n_ij) * n_ij|
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param v Velocity vector
     * @param n_ij Unit vector from i to j
     * @return Tangent velocity magnitude
     */
    template<class Tscal, class Tvec>
    inline Tscal sr_tangent_velocity_mag(Tvec v, Tvec n_ij) {
        Tvec v_t_vec = sr_tangent_velocity_vec<Tscal, Tvec>(v, n_ij);
        return sycl::sqrt(sycl::dot(v_t_vec, v_t_vec));
    }

    /**
     * @brief Full SR force computation for a particle pair (interpolated V² formula)
     *
     * Computes both momentum and energy contributions in one call.
     * Uses Kitajima et al. (2025) Eq. 371-372 with interpolated V²:
     *   dS_i/dt = -P* * V²_interp * (∇W_i + ∇W_j)
     *   de_i/dt = v* · dS_i/dt
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     * @param P_star Interface pressure
     * @param v_x_star Interface normal velocity
     * @param v_t_star Interface tangent velocity
     * @param n_ij Unit vector from i to j
     * @param v_t_dir_L Left tangent velocity direction (unit vector)
     * @param v_t_dir_R Right tangent velocity direction (unit vector)
     * @param V_i Volume of particle i
     * @param V_j Volume of particle j
     * @param grad_W_i Kernel gradient for particle i
     * @param grad_W_j Kernel gradient for particle j
     * @param dS_i Output: momentum derivative contribution
     * @param de_i Output: energy derivative contribution
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

        // Compute momentum force using Kitajima interpolated V² formula
        Tvec force = sr_momentum_force<Tscal, Tvec>(P_star, V_i, V_j, grad_W_i, grad_W_j);

        // Reconstruct v* vector (use upwind direction for tangent)
        Tvec v_t_dir = (v_x_star > Tscal{0}) ? v_t_dir_L : v_t_dir_R;
        Tvec v_star  = n_ij * v_x_star + v_t_dir * v_t_star;

        // Compute energy evolution
        dS_i = force;
        de_i = sr_energy_evolution<Tscal, Tvec>(v_star, force);
    }

    /**
     * @brief Full SR force computation with grad-h correction
     *
     * Uses grad-h corrected force formula:
     *   dS_i/dt = -P* * (V_i²/Ω_i * ∇W_i + V_j²/Ω_j * ∇W_j)
     *   de_i/dt = v* · dS_i/dt
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
        Tscal omega_i,
        Tscal omega_j,
        Tvec grad_W_i,
        Tvec grad_W_j,
        Tvec &dS_i,
        Tscal &de_i) {

        // Compute momentum force using grad-h corrected V²/Ω formula
        Tvec force =
            sr_momentum_force<Tscal, Tvec>(P_star, V_i, V_j, omega_i, omega_j, grad_W_i, grad_W_j);

        // Reconstruct v* vector (use upwind direction for tangent)
        Tvec v_t_dir = (v_x_star > Tscal{0}) ? v_t_dir_L : v_t_dir_R;
        Tvec v_star  = n_ij * v_x_star + v_t_dir * v_t_star;

        // Compute energy evolution
        dS_i = force;
        de_i = sr_energy_evolution<Tscal, Tvec>(v_star, force);
    }

    /**
     * @brief Simplified SR force computation (no tangent velocity)
     *
     * For cases where tangent velocity is zero or not needed.
     * Uses Kitajima interpolated V² formulation.
     *
     * @tparam Tscal Scalar type
     * @tparam Tvec Vector type
     */
    template<class Tscal, class Tvec>
    inline void sr_pairwise_force_simple(
        Tscal P_star,
        Tscal v_x_star,
        Tvec n_ij,
        Tscal V_i,
        Tscal V_j,
        Tvec grad_W_i,
        Tvec grad_W_j,
        Tvec &dS_i,
        Tscal &de_i) {

        // Compute momentum force using Kitajima interpolated V² formula
        Tvec force = sr_momentum_force<Tscal, Tvec>(P_star, V_i, V_j, grad_W_i, grad_W_j);

        // v* is purely normal direction
        Tvec v_star = n_ij * v_x_star;

        // Compute energy evolution
        dS_i = force;
        de_i = sr_energy_evolution<Tscal, Tvec>(v_star, force);
    }

    /**
     * @brief Simplified SR force computation with grad-h correction
     */
    template<class Tscal, class Tvec>
    inline void sr_pairwise_force_simple(
        Tscal P_star,
        Tscal v_x_star,
        Tvec n_ij,
        Tscal V_i,
        Tscal V_j,
        Tscal omega_i,
        Tscal omega_j,
        Tvec grad_W_i,
        Tvec grad_W_j,
        Tvec &dS_i,
        Tscal &de_i) {

        // Compute momentum force using grad-h corrected V²/Ω formula
        Tvec force =
            sr_momentum_force<Tscal, Tvec>(P_star, V_i, V_j, omega_i, omega_j, grad_W_i, grad_W_j);

        // v* is purely normal direction
        Tvec v_star = n_ij * v_x_star;

        // Compute energy evolution
        dS_i = force;
        de_i = sr_energy_evolution<Tscal, Tvec>(v_star, force);
    }

} // namespace shammodels::gsph::sr
