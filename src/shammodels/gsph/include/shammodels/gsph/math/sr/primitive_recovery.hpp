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
 * @file primitive_recovery.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Primitive variable recovery for Special Relativistic GSPH
 *
 * Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 *
 * Converts between conserved variables (S, e, N) and primitive variables (v, ρ, P).
 * The key challenge is solving the quartic equation for the Lorentz factor γ.
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::sr {

    /**
     * @brief Primitive variables for SR-GSPH
     */
    template<class Tscal>
    struct SRPrimitiveVars {
        Tscal gamma_lor;   ///< Lorentz factor γ
        Tscal density;     ///< Rest-frame density n
        Tscal pressure;    ///< Pressure P
        Tscal enthalpy;    ///< Specific enthalpy H
        Tscal sound_speed; ///< Sound speed c_s
        Tscal vel_normal;  ///< Normal velocity component v_x
        Tscal vel_tangent; ///< Tangent velocity component v_t
    };

    /**
     * @brief Solve quartic equation for Lorentz factor using Newton-Raphson
     *
     * Equation: (γ²-1)(Xeγ-1)² - S²(Xγ²-1)² = 0
     * where X = γ_eos/(γ_eos-1)
     *
     * @param S_mag Magnitude of canonical momentum |S|
     * @param e Canonical energy per baryon
     * @param gamma_eos Adiabatic index γ_c
     * @param max_iter Maximum Newton-Raphson iterations
     * @param tol Convergence tolerance
     * @return Lorentz factor γ
     */
    template<class Tscal>
    inline Tscal solve_lorentz_factor(
        Tscal S_mag, Tscal e, Tscal gamma_eos, u32 max_iter = 50, Tscal tol = Tscal{1e-10}) {

        const Tscal X  = gamma_eos / (gamma_eos - Tscal{1});
        const Tscal S2 = S_mag * S_mag;

        // Quartic function f(γ) = (γ²-1)(Xeγ-1)² - S²(Xγ²-1)²
        auto func = [&](Tscal gamma) -> Tscal {
            const Tscal A     = X * e * gamma - Tscal{1};
            const Tscal B     = X * gamma * gamma - Tscal{1};
            const Tscal term1 = (gamma * gamma - Tscal{1}) * A * A;
            const Tscal term2 = S2 * B * B;
            return term1 - term2;
        };

        // Derivative df/dγ
        auto dfunc = [&](Tscal gamma) -> Tscal {
            const Tscal term1_A = gamma * gamma - Tscal{1};
            const Tscal term1_B = X * e * gamma - Tscal{1};
            const Tscal d_term1
                = Tscal{2} * gamma * term1_B * term1_B + term1_A * Tscal{2} * term1_B * X * e;

            const Tscal term2_A = X * gamma * gamma - Tscal{1};
            const Tscal d_term2 = S2 * Tscal{2} * term2_A * Tscal{2} * X * gamma;

            return d_term1 - d_term2;
        };

        // Initial guess: v ≈ S/e, then γ = 1/√(1-v²)
        Tscal v_guess = Tscal{0};
        if (e > Tscal{0}) {
            v_guess = S_mag / e;
            if (v_guess >= Tscal{1}) {
                v_guess = Tscal{0.99};
            }
        }
        Tscal gamma = Tscal{1} / sycl::sqrt(Tscal{1} - v_guess * v_guess);

        // Newton-Raphson iteration
        for (u32 iter = 0; iter < max_iter; ++iter) {
            const Tscal f  = func(gamma);
            const Tscal df = dfunc(gamma);

            if (sycl::fabs(df) < Tscal{1e-15}) {
                break;
            }

            const Tscal delta = f / df;
            gamma -= delta;

            // Safety: γ ≥ 1
            if (gamma < Tscal{1}) {
                gamma = Tscal{1} + Tscal{1e-10};
            }

            if (sycl::fabs(delta) < tol) {
                break;
            }
        }

        return sycl::fmax(gamma, Tscal{1});
    }

    /**
     * @brief Convert conserved to primitive variables with tangent momentum
     *
     * Uses conserved tangent momentum S_t (dS_t/dt = 0 in 1D).
     * Recovers v_t = S_t / (γH) at each step.
     *
     * IMPORTANT: Takes LAB-FRAME density N as input (from SPH summation).
     * Converts to rest-frame density n = N / γ after solving for Lorentz factor.
     *
     * @param S_normal_mag Magnitude of normal canonical momentum |S_x|
     * @param S_t Tangent canonical momentum (conserved)
     * @param e Canonical energy per baryon
     * @param N Lab-frame density (SPH summation gives N = ν·ΣW, NOT rest-frame n)
     * @param gamma_eos Adiabatic index γ_c
     * @param c_speed Speed of light
     * @return Primitive variables
     */
    template<class Tscal>
    inline SRPrimitiveVars<Tscal> conserved_to_primitive(
        Tscal S_normal_mag, Tscal S_t, Tscal e, Tscal N, Tscal gamma_eos, Tscal c_speed) {

        SRPrimitiveVars<Tscal> prim;

        const Tscal X       = gamma_eos / (gamma_eos - Tscal{1});
        const Tscal c2      = c_speed * c_speed;
        const Tscal S_t_abs = sycl::fabs(S_t);

        // Total momentum magnitude for initial γ estimate
        const Tscal S_total = sycl::sqrt(S_normal_mag * S_normal_mag + S_t_abs * S_t_abs);
        Tscal gamma_lor     = solve_lorentz_factor(S_total, e, gamma_eos);

        // Compute H from γ
        Tscal denom_H = X * gamma_lor * gamma_lor - Tscal{1};
        Tscal H       = (denom_H > Tscal{1e-10}) ? (X * e * gamma_lor - Tscal{1}) / denom_H
                                                 : Tscal{1} + Tscal{1e-8};
        H             = sycl::fmax(H, Tscal{1} + Tscal{1e-8});

        // Extract velocities from S = γHv and S_t = γHv_t
        Tscal gamma_H = gamma_lor * H;
        Tscal v_x     = (gamma_H > Tscal{1e-10}) ? S_normal_mag / gamma_H : Tscal{0};
        Tscal v_t     = (gamma_H > Tscal{1e-10}) ? S_t / gamma_H : Tscal{0};

        // Iterative refinement for consistency with γ constraint
        for (u32 iter = 0; iter < 10; ++iter) {
            Tscal v2 = v_x * v_x + v_t * v_t;

            // Clamp to subluminal
            if (v2 >= Tscal{0.9999}) {
                const Tscal scale = sycl::sqrt(Tscal{0.99} / v2);
                v_x *= scale;
                v_t *= scale;
                v2 = v_x * v_x + v_t * v_t;
            }

            // Update γ and H
            const Tscal gamma_new = Tscal{1} / sycl::sqrt(Tscal{1} - v2);
            denom_H               = X * gamma_new * gamma_new - Tscal{1};
            if (denom_H < Tscal{1e-10}) {
                denom_H = Tscal{1e-10};
            }
            const Tscal H_new
                = sycl::fmax((X * e * gamma_new - Tscal{1}) / denom_H, Tscal{1} + Tscal{1e-8});

            // Re-extract velocities
            gamma_H             = gamma_new * H_new;
            const Tscal v_x_new = S_normal_mag / gamma_H;
            const Tscal v_t_new = S_t / gamma_H;

            // Check convergence
            if (sycl::fabs(gamma_new - gamma_lor) / gamma_lor < Tscal{1e-8}
                && sycl::fabs(v_t_new - v_t) < Tscal{1e-8}) {
                gamma_lor = gamma_new;
                H         = H_new;
                v_x       = v_x_new;
                v_t       = v_t_new;
                break;
            }

            gamma_lor = gamma_new;
            H         = H_new;
            v_x       = v_x_new;
            v_t       = v_t_new;
        }

        prim.gamma_lor   = gamma_lor;
        prim.vel_normal  = v_x;
        prim.vel_tangent = v_t;
        prim.enthalpy    = H;

        // Rest-frame density: convert from lab-frame N to rest-frame n = N / γ
        prim.density = N / sycl::fmax(gamma_lor, Tscal{1});

        // Pressure: P = n(H-1)(γ_c-1)/γ_c
        prim.pressure = prim.density * (H - Tscal{1}) * (gamma_eos - Tscal{1}) / gamma_eos;
        prim.pressure = sycl::fmax(prim.pressure, Tscal{1e-6});

        // Sound speed: c_s² = (γ_c-1)(H-1)/H
        const Tscal cs2  = (gamma_eos - Tscal{1}) * (H - Tscal{1}) / H;
        prim.sound_speed = sycl::sqrt(sycl::fmax(cs2, Tscal{0})) * c_speed;

        return prim;
    }

    /**
     * @brief Convert primitive to conserved variables
     *
     * S = γHv
     * e = γH - P/(Nc²)
     *
     * @param v_x Normal velocity component
     * @param v_t Tangent velocity component
     * @param density Rest-frame density n
     * @param pressure Pressure P
     * @param N Lab-frame baryon density
     * @param gamma_eos Adiabatic index
     * @param c_speed Speed of light
     * @param S_out Output: normal canonical momentum
     * @param S_t_out Output: tangent canonical momentum
     * @param e_out Output: canonical energy
     */
    template<class Tscal>
    inline void primitive_to_conserved(
        Tscal v_x,
        Tscal v_t,
        Tscal density,
        Tscal pressure,
        Tscal N,
        Tscal gamma_eos,
        Tscal c_speed,
        Tscal &S_out,
        Tscal &S_t_out,
        Tscal &e_out) {

        const Tscal c2 = c_speed * c_speed;
        const Tscal v2 = v_x * v_x + v_t * v_t;

        // Lorentz factor
        const Tscal gamma_lor = Tscal{1} / sycl::sqrt(Tscal{1} - v2 / c2);

        // Internal energy per unit mass
        const Tscal u = pressure / ((gamma_eos - Tscal{1}) * density);

        // Specific enthalpy: H = 1 + u/c² + P/(nc²)
        const Tscal H = Tscal{1} + u / c2 + pressure / (density * c2);

        // Canonical momentum: S = γHv
        S_out   = gamma_lor * H * v_x;
        S_t_out = gamma_lor * H * v_t;

        // Canonical energy: e = γH - P/(Nc²)
        e_out = gamma_lor * H - pressure / (N * c2);
    }

} // namespace shammodels::gsph::sr
