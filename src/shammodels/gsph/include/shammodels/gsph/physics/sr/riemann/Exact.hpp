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
 * @file Exact.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Exact Riemann Solver for Special Relativistic Hydrodynamics
 *
 * Implements the exact iterative Riemann solver for SR hydro following:
 * - Pons et al. (2000) "The exact solution of the Riemann problem with
 *   non-zero tangential velocities in relativistic hydrodynamics"
 * - Kitajima et al. (2025) Godunov SPH with exact Riemann solver
 *
 * The solver uses hybrid Newton-bisection to find P* by solving:
 *   v_x_L(P*) - v_x_R(P*) = 0
 *
 * Where v_x is computed from:
 * - Shock: Pons et al. (2000) Eq. 25-27 jump conditions
 * - Rarefaction: Riemann invariant integration with arctanh transformation
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/gsph/physics/sr/riemann/RiemannBase.hpp"

namespace shammodels::gsph::physics::sr::riemann {

    // =========================================================================
    // Gauss-Legendre quadrature (16-point) for rarefaction integration
    // =========================================================================
    static constexpr int GAUSS_POINTS = 16;

    template<class Tscal>
    struct GaussLegendre {
        static constexpr Tscal nodes[GAUSS_POINTS]
            = {Tscal{-0.9894009349916499},
               Tscal{-0.9445750230732326},
               Tscal{-0.8656312023878318},
               Tscal{-0.7554044083550030},
               Tscal{-0.6178762444026438},
               Tscal{-0.4580167776572274},
               Tscal{-0.2816035507792589},
               Tscal{-0.0950125098376374},
               Tscal{0.0950125098376374},
               Tscal{0.2816035507792589},
               Tscal{0.4580167776572274},
               Tscal{0.6178762444026438},
               Tscal{0.7554044083550030},
               Tscal{0.8656312023878318},
               Tscal{0.9445750230732326},
               Tscal{0.9894009349916499}};

        static constexpr Tscal weights[GAUSS_POINTS]
            = {Tscal{0.0271524594117541},
               Tscal{0.0622535239386479},
               Tscal{0.0951585116824928},
               Tscal{0.1246289712555339},
               Tscal{0.1495959888165767},
               Tscal{0.1691565193950025},
               Tscal{0.1826034150449236},
               Tscal{0.1894506104550685},
               Tscal{0.1894506104550685},
               Tscal{0.1826034150449236},
               Tscal{0.1691565193950025},
               Tscal{0.1495959888165767},
               Tscal{0.1246289712555339},
               Tscal{0.0951585116824928},
               Tscal{0.0622535239386479},
               Tscal{0.0271524594117541}};
    };

    // =========================================================================
    // Isentropic state for rarefaction waves
    // =========================================================================
    template<class Tscal>
    struct IsentropeState {
        Tscal K;       // Entropy constant K = P / n^gamma
        Tscal gamma_c; // Adiabatic index

        inline IsentropeState(Tscal P, Tscal n, Tscal gamma) : gamma_c(gamma) {
            K = P / sycl::pow(sycl::fmax(n, Tscal{1e-15}), gamma);
        }

        inline Tscal compute_rho(Tscal P) const {
            return sycl::pow(sycl::fmax(P, Tscal{1e-15}) / K, Tscal{1} / gamma_c);
        }

        inline Tscal compute_enthalpy(Tscal P) const {
            const Tscal rho = compute_rho(P);
            return Tscal{1} + (gamma_c / (gamma_c - Tscal{1})) * P / rho;
        }

        inline Tscal compute_enthalpy(Tscal P, Tscal rho) const {
            return Tscal{1} + (gamma_c / (gamma_c - Tscal{1})) * P / rho;
        }

        inline Tscal compute_sound_speed(Tscal P, Tscal rho, Tscal h) const {
            return sycl::sqrt(gamma_c * P / (rho * h));
        }
    };

    // =========================================================================
    // Helper: compute v_t from conserved invariant A = h * W * v_t
    // =========================================================================
    template<class Tscal>
    inline Tscal compute_vt_from_invariant(Tscal A, Tscal v_x, Tscal h) {
        if (sycl::fabs(A) < Tscal{1e-15}) {
            return Tscal{0};
        }
        const Tscal one_minus_vx2 = sycl::fmax(Tscal{0}, Tscal{1} - v_x * v_x);
        const Tscal denom         = h * h + A * A;
        if (denom < Tscal{1e-30}) {
            return Tscal{0};
        }
        return A * sycl::sqrt(one_minus_vx2 / denom);
    }

    // =========================================================================
    // Shock solver: Pons et al. (2000) Eq. 25-27
    // =========================================================================
    template<class Tscal>
    inline bool solve_shock(
        Tscal P_star,
        Tscal P_a,
        Tscal n_a,
        Tscal v_x_a,
        Tscal v_t_a,
        Tscal gamma_c,
        bool is_left_wave,
        Tscal &v_x_b,
        Tscal &n_b,
        Tscal &H_b,
        Tscal &v_t_b) {

        // Initialize outputs to sensible defaults (in case of early return)
        v_x_b = v_x_a;
        v_t_b = v_t_a;
        n_b   = n_a;
        H_b   = Tscal{1} + (gamma_c / (gamma_c - Tscal{1})) * (P_a / n_a);

        if (P_star <= Tscal{0}) {
            return false;
        }

        const Tscal P_a_safe = sycl::fmax(P_a, Tscal{1e-10});
        const Tscal P_b      = P_star;

        // Enthalpy ahead: H = 1 + (gamma/(gamma-1)) * P/n
        const Tscal H_a = Tscal{1} + (gamma_c / (gamma_c - Tscal{1})) * (P_a_safe / n_a);

        // Quadratic for H_b (Pons et al. Eq. 27)
        const Tscal A_coef = (gamma_c - Tscal{1}) * (P_a_safe - P_b) / (gamma_c * P_b);
        const Tscal B_coef = H_a * (P_a_safe - P_b) / n_a;

        const Tscal q_a = Tscal{1} + A_coef;
        const Tscal q_b = -A_coef;
        const Tscal q_c = B_coef - H_a * H_a;

        const Tscal discriminant = q_b * q_b - Tscal{4} * q_a * q_c;
        if (discriminant < Tscal{0}) {
            return false;
        }

        H_b = (-q_b + sycl::sqrt(discriminant)) / (Tscal{2} * q_a);
        // Safety: ensure H_b > 1 (minimum enthalpy)
        H_b = sycl::fmax(H_b, Tscal{1} + Tscal{1e-10});

        // Density behind shock from EOS
        n_b = (gamma_c / (gamma_c - Tscal{1})) * P_b / (H_b - Tscal{1});
        // Safety: floor density
        n_b = sycl::fmax(n_b, Tscal{1e-15});

        // Mass flux j^2 (Pons et al.)
        const Tscal denom_j = (H_b / n_b - H_a / n_a);
        if (sycl::fabs(denom_j) < Tscal{1e-15}) {
            v_x_b = v_x_a;
            v_t_b = v_t_a;
            return true;
        }

        const Tscal j2 = -(P_b - P_a_safe) / denom_j;
        if (j2 <= Tscal{0}) {
            v_x_b = v_x_a;
            v_t_b = v_t_a;
            return true;
        }

        const Tscal j = sycl::sqrt(j2);

        // Lorentz factor - MUST include tangential velocity
        // Safety: clamp total velocity to subluminal
        const Tscal v2_total_a = sycl::fmin(v_x_a * v_x_a + v_t_a * v_t_a, Tscal{0.9999});
        const Tscal gamma_a    = Tscal{1} / sycl::sqrt(Tscal{1} - v2_total_a);
        const Tscal N_a        = n_a * gamma_a; // Lab-frame density

        // Shock speed V_s (Pons et al. Eq. 26)
        const Tscal term      = j * j + N_a * N_a * (Tscal{1} - v_x_a * v_x_a);
        const Tscal sqrt_term = sycl::sqrt(sycl::fmax(term, Tscal{0}));
        const Tscal denom_Vs  = N_a * N_a + j * j;

        Tscal V_s;
        if (is_left_wave) {
            V_s = (N_a * N_a * v_x_a - j * sqrt_term) / denom_Vs;
        } else {
            V_s = (N_a * N_a * v_x_a + j * sqrt_term) / denom_Vs;
        }
        // Safety: clamp shock speed to subluminal
        V_s = sycl::fmax(Tscal{-0.9999}, sycl::fmin(V_s, Tscal{0.9999}));

        const Tscal gamma_s  = Tscal{1} / sycl::sqrt(Tscal{1} - V_s * V_s);
        const Tscal j_signed = is_left_wave ? -j : j;

        if (sycl::fabs(j_signed) < Tscal{1e-15}) {
            v_x_b = v_x_a;
            v_t_b = v_t_a;
            return true;
        }

        // Normal velocity behind shock (Pons et al. Eq. 25)
        const Tscal num = H_a * gamma_a * v_x_a + gamma_s * (P_b - P_a_safe) / j_signed;
        const Tscal den
            = H_a * gamma_a + (P_b - P_a_safe) * (gamma_s * v_x_a / j_signed + Tscal{1} / N_a);

        if (sycl::fabs(den) < Tscal{1e-15}) {
            return false;
        }

        v_x_b = num / den;
        // Safety: clamp output velocity to subluminal
        v_x_b = sycl::fmax(Tscal{-0.9999}, sycl::fmin(v_x_b, Tscal{0.9999}));

        // Tangential velocity (Pons et al. Eq. 25)
        if (sycl::fabs(v_t_a) > Tscal{1e-10}) {
            const Tscal factor_num = sycl::fmax(Tscal{0}, Tscal{1} - v_x_b * v_x_b);
            const Tscal factor_den = H_b * H_b + (H_a * gamma_a * v_t_a) * (H_a * gamma_a * v_t_a);
            if (factor_num > Tscal{0} && factor_den > Tscal{0}) {
                v_t_b = v_t_a * H_a * gamma_a * sycl::sqrt(factor_num / factor_den);
                // Safety: ensure total velocity stays subluminal
                const Tscal v2_total = v_x_b * v_x_b + v_t_b * v_t_b;
                if (v2_total >= Tscal{0.9999}) {
                    const Tscal scale = sycl::sqrt(Tscal{0.99} / v2_total);
                    v_t_b *= scale;
                }
            } else {
                v_t_b = Tscal{0};
            }
        } else {
            v_t_b = Tscal{0};
        }

        return true;
    }

    // =========================================================================
    // Rarefaction solver (zero tangent velocity - analytical)
    // =========================================================================
    template<class Tscal>
    inline bool solve_rarefaction_zero_tangent(
        Tscal P_star,
        Tscal P_a,
        Tscal n_a,
        Tscal v_x_a,
        Tscal gamma_c,
        bool is_left_wave,
        Tscal &v_x_b,
        Tscal &n_b,
        Tscal &H_b,
        Tscal &v_t_b) {

        const Tscal P_a_safe = sycl::fmax(P_a, Tscal{1e-10});

        // Isentropic relation: P/n^gamma = const
        const Tscal K_entropy = P_a_safe / sycl::pow(n_a, gamma_c);
        n_b                   = sycl::pow(P_star / K_entropy, Tscal{1} / gamma_c);

        // Sound speeds
        const Tscal u_a   = P_a_safe / ((gamma_c - Tscal{1}) * n_a);
        const Tscal H_a   = Tscal{1} + u_a + P_a_safe / n_a;
        const Tscal c_s_a = sycl::sqrt(gamma_c * P_a_safe / (n_a * H_a));

        const Tscal u_b   = P_star / ((gamma_c - Tscal{1}) * n_b);
        H_b               = Tscal{1} + u_b + P_star / n_b;
        const Tscal c_s_b = sycl::sqrt(gamma_c * P_star / (n_b * H_b));

        // Riemann invariant (analytical for v_t = 0)
        const Tscal sign     = is_left_wave ? Tscal{1} : Tscal{-1};
        const Tscal sqrt_gm1 = sycl::sqrt(gamma_c - Tscal{1});

        // Safety: cap c_s to avoid singularity when c_s → sqrt(γ-1)
        // In ultra-relativistic limit, c_s → sqrt((γ-1)/γ) < sqrt(γ-1), but
        // numerical errors can push c_s close to sqrt_gm1
        const Tscal c_s_max    = sqrt_gm1 * Tscal{0.999};
        const Tscal c_s_a_safe = sycl::fmin(c_s_a, c_s_max);
        const Tscal c_s_b_safe = sycl::fmin(c_s_b, c_s_max);

        // Clamp v_x_a to avoid singularity in term_v
        const Tscal v_x_a_safe = sycl::fmax(Tscal{-0.9999}, sycl::fmin(Tscal{0.9999}, v_x_a));

        const Tscal term_v   = (Tscal{1} + v_x_a_safe) / (Tscal{1} - v_x_a_safe);
        const Tscal term_c_a = (sqrt_gm1 + c_s_a_safe) / (sqrt_gm1 - c_s_a_safe);
        const Tscal term_c_b = (sqrt_gm1 + c_s_b_safe) / (sqrt_gm1 - c_s_b_safe);

        const Tscal exponent = sign * Tscal{2} / sqrt_gm1;
        const Tscal base     = term_c_a / term_c_b;

        // Safety: limit base to prevent overflow in pow
        const Tscal base_safe = sycl::fmax(Tscal{1e-10}, sycl::fmin(base, Tscal{1e10}));
        const Tscal A_val     = term_v * sycl::pow(base_safe, exponent);
        v_x_b                 = (A_val - Tscal{1}) / (A_val + Tscal{1});

        v_t_b = Tscal{0};

        return true;
    }

    // =========================================================================
    // Rarefaction integrand for Gauss-Legendre quadrature
    // integrand(P) = sqrt(h^2 + A^2(1 - c_s^2)) / ((h^2 + A^2) * rho * c_s)
    // =========================================================================
    template<class Tscal>
    inline Tscal rarefaction_integrand(
        Tscal P, Tscal A_sqr, const IsentropeState<Tscal> &isentrope) {
        const Tscal P_safe = sycl::fmax(P, Tscal{1e-15});
        const Tscal rho    = isentrope.compute_rho(P_safe);
        const Tscal h      = isentrope.compute_enthalpy(P_safe, rho);
        const Tscal cs     = isentrope.compute_sound_speed(P_safe, rho, h);

        if (cs < Tscal{1e-15} || rho < Tscal{1e-15}) {
            return Tscal{0}; // Return 0 instead of NaN for GPU safety
        }

        const Tscal h_sqr  = h * h;
        const Tscal cs_sqr = cs * cs;

        // Integrand = sqrt(h^2 + A^2(1 - c_s^2)) / ((h^2 + A^2) * rho * c_s)
        const Tscal numerator_sqr = h_sqr + A_sqr * (Tscal{1} - cs_sqr);
        if (numerator_sqr < Tscal{0}) {
            return Tscal{0};
        }

        const Tscal numerator   = sycl::sqrt(numerator_sqr);
        const Tscal denominator = (h_sqr + A_sqr) * rho * cs;

        if (denominator < Tscal{1e-30}) {
            return Tscal{0};
        }

        return numerator / denominator;
    }

    // =========================================================================
    // Rarefaction solver (with tangent velocity - numerical integration)
    // Uses arctanh transformation: v_x_b = tanh(B) where
    // B = arctanh(v_x_a) + sign * integral
    // =========================================================================
    template<class Tscal>
    inline bool solve_rarefaction(
        Tscal P_star,
        Tscal P_a,
        Tscal n_a,
        Tscal v_x_a,
        Tscal v_t_a,
        Tscal gamma_c,
        bool is_left_wave,
        Tscal &v_x_b,
        Tscal &n_b,
        Tscal &H_b,
        Tscal &v_t_b) {

        // Initialize outputs to sensible defaults (in case of early return)
        v_x_b = v_x_a;
        v_t_b = v_t_a;
        n_b   = n_a;
        H_b   = Tscal{1} + (gamma_c / (gamma_c - Tscal{1})) * (P_a / n_a);

        const Tscal P_a_safe = sycl::fmax(P_a, Tscal{1e-10});

        // Use analytical solution for zero tangent velocity
        if (sycl::fabs(v_t_a) < Tscal{1e-10}) {
            return solve_rarefaction_zero_tangent(
                P_star, P_a_safe, n_a, v_x_a, gamma_c, is_left_wave, v_x_b, n_b, H_b, v_t_b);
        }

        if (P_star > P_a_safe) {
            return false; // Should use shock solver
        }

        // Create isentrope from initial state
        const IsentropeState<Tscal> isentrope(P_a_safe, n_a, gamma_c);

        // Conserved invariant A = h * W * v_t
        const Tscal H_a        = isentrope.compute_enthalpy(P_a_safe);
        const Tscal v2_total_a = v_x_a * v_x_a + v_t_a * v_t_a;
        const Tscal v2_clamped = sycl::fmin(v2_total_a, Tscal{1} - Tscal{1e-12});
        const Tscal W_a        = Tscal{1} / sycl::sqrt(Tscal{1} - v2_clamped);
        const Tscal A          = H_a * W_a * v_t_a;
        const Tscal A_sqr      = A * A;

        // Sign convention: -1 for left wave, +1 for right wave
        const Tscal sign = is_left_wave ? Tscal{-1} : Tscal{1};

        // arctanh(v_x_a) = 0.5 * ln((1+v_x)/(1-v_x))
        const Tscal v_x_a_clamped = sycl::fmax(Tscal{-0.9999}, sycl::fmin(Tscal{0.9999}, v_x_a));
        const Tscal arctanh_vxa
            = Tscal{0.5} * sycl::log((Tscal{1} + v_x_a_clamped) / (Tscal{1} - v_x_a_clamped));

        // Gauss-Legendre integration from P_a to P_star
        const Tscal P_mid   = Tscal{0.5} * (P_a_safe + P_star);
        const Tscal half_dP = Tscal{0.5} * (P_star - P_a_safe);

        Tscal integral = Tscal{0};
        for (int i = 0; i < GAUSS_POINTS; ++i) {
            const Tscal t = GaussLegendre<Tscal>::nodes[i];
            const Tscal P = P_mid + half_dP * t;
            const Tscal f = rarefaction_integrand(P, A_sqr, isentrope);
            integral += GaussLegendre<Tscal>::weights[i] * f;
        }
        integral *= half_dP; // Jacobian

        // B = arctanh(v_x_a) + sign * integral
        const Tscal B = arctanh_vxa + sign * integral;

        // v_x_b = tanh(B)
        v_x_b = sycl::tanh(B);
        v_x_b = sycl::fmax(Tscal{-0.9999}, sycl::fmin(Tscal{0.9999}, v_x_b));

        // Final state from isentropic relations
        n_b = isentrope.compute_rho(P_star);
        H_b = isentrope.compute_enthalpy(P_star, n_b);

        // Tangent velocity from conserved invariant
        v_t_b = compute_vt_from_invariant(A, v_x_b, H_b);

        return true;
    }

    // =========================================================================
    // Wave curve: v_x as function of P for root finding
    // =========================================================================
    template<class Tscal>
    inline Tscal wave_curve(
        Tscal P,
        Tscal P_state,
        Tscal n_state,
        Tscal v_x_state,
        Tscal v_t_state,
        Tscal gamma_c,
        bool is_left_wave) {

        Tscal v_x_star, n_star, H_star, v_t_star;
        bool success;

        if (P > P_state) {
            success = solve_shock(
                P,
                P_state,
                n_state,
                v_x_state,
                v_t_state,
                gamma_c,
                is_left_wave,
                v_x_star,
                n_star,
                H_star,
                v_t_star);
        } else {
            success = solve_rarefaction(
                P,
                P_state,
                n_state,
                v_x_state,
                v_t_state,
                gamma_c,
                is_left_wave,
                v_x_star,
                n_star,
                H_star,
                v_t_star);
        }

        // Return large value on failure to help bracketing
        return success ? v_x_star : (is_left_wave ? Tscal{-10} : Tscal{10});
    }

    // =========================================================================
    // Exact SR Riemann solver using hybrid Newton-bisection
    // =========================================================================
    template<class Tscal>
    inline Result<Tscal> solve(
        Tscal v_x_L,
        Tscal v_t_L,
        Tscal n_L,
        Tscal P_L,
        Tscal v_x_R,
        Tscal v_t_R,
        Tscal n_R,
        Tscal P_R,
        Tscal gamma,
        Tscal tol    = Tscal{1e-10},
        u32 max_iter = 100) {

        Result<Tscal> result;
        result.converged = false;

        // Safety floors
        n_L = sycl::fmax(n_L, Tscal{1e-15});
        n_R = sycl::fmax(n_R, Tscal{1e-15});
        P_L = sycl::fmax(P_L, Tscal{1e-15});
        P_R = sycl::fmax(P_R, Tscal{1e-15});

        // Check for identical states
        if (sycl::fabs(P_L - P_R) < Tscal{1e-10} && sycl::fabs(n_L - n_R) < Tscal{1e-10}
            && sycl::fabs(v_x_L - v_x_R) < Tscal{1e-10}) {
            result.P_star    = P_L;
            result.v_x_star  = v_x_L;
            result.v_t_star  = v_t_L;
            result.converged = true;
            return result;
        }

        // Clamp superluminal velocities (like reference Python implementation)
        Tscal v2_L             = v_x_L * v_x_L + v_t_L * v_t_L;
        Tscal v2_R             = v_x_R * v_x_R + v_t_R * v_t_R;
        constexpr Tscal v2_max = Tscal{0.99999999};
        if (v2_L >= v2_max) {
            const Tscal factor = sycl::sqrt(v2_max / v2_L);
            v_x_L *= factor;
            v_t_L *= factor;
            v2_L = v2_max;
        }
        if (v2_R >= v2_max) {
            const Tscal factor = sycl::sqrt(v2_max / v2_R);
            v_x_R *= factor;
            v_t_R *= factor;
            v2_R = v2_max;
        }

        // Residual function: f(P) = v_x_L(P) - v_x_R(P)
        auto f = [&](Tscal P) -> Tscal {
            const Tscal v_x_L_star = wave_curve(P, P_L, n_L, v_x_L, v_t_L, gamma, true);
            const Tscal v_x_R_star = wave_curve(P, P_R, n_R, v_x_R, v_t_R, gamma, false);
            return v_x_L_star - v_x_R_star;
        };

        // Helper to check if value is valid (finite)
        auto is_valid = [](Tscal x) {
            return sycl::isfinite(x);
        };

        // Initial bracket
        const Tscal P_min = sycl::fmin(P_L, P_R);
        const Tscal P_max = sycl::fmax(P_L, P_R);
        Tscal P_lo        = P_min * Tscal{1e-6};
        Tscal P_hi        = P_max * Tscal{1e6};

        Tscal f_lo = f(P_lo);
        Tscal f_hi = f(P_hi);

        // If NaN in initial bracket, try smaller range
        if (!is_valid(f_lo) || !is_valid(f_hi)) {
            P_lo = P_min * Tscal{0.01};
            P_hi = P_max * Tscal{100};
            f_lo = f(P_lo);
            f_hi = f(P_hi);
        }

        // If still NaN, return fallback
        if (!is_valid(f_lo) || !is_valid(f_hi)) {
            result.P_star    = Tscal{0.5} * (P_L + P_R);
            result.v_x_star  = Tscal{0.5} * (v_x_L + v_x_R);
            result.v_t_star  = Tscal{0};
            result.converged = false;
            return result;
        }

        // If not bracketed (same sign), try smaller range
        if (f_lo * f_hi > Tscal{0}) {
            P_lo = P_min * Tscal{0.01};
            P_hi = P_max * Tscal{100};
            f_lo = f(P_lo);
            f_hi = f(P_hi);
        }

        // If still not bracketed, search for bracket
        if (f_lo * f_hi > Tscal{0} || !is_valid(f_lo) || !is_valid(f_hi)) {
            const Tscal log_P_lo = sycl::log10(P_min * Tscal{1e-6});
            const Tscal log_P_hi = sycl::log10(P_max * Tscal{1e6});

            Tscal prev_P = P_lo;
            Tscal prev_f = f_lo;
            bool found   = false;

            for (int i = 1; i <= 20 && !found; ++i) {
                const Tscal log_P  = log_P_lo + (log_P_hi - log_P_lo) * Tscal(i) / Tscal{20};
                const Tscal curr_P = sycl::pow(Tscal{10}, log_P);
                const Tscal curr_f = f(curr_P);

                // Only use valid values for bracketing (like reference)
                if (is_valid(curr_f) && is_valid(prev_f) && prev_f * curr_f < Tscal{0}) {
                    P_lo  = prev_P;
                    P_hi  = curr_P;
                    f_lo  = prev_f;
                    f_hi  = curr_f;
                    found = true;
                }
                if (is_valid(curr_f)) {
                    prev_P = curr_P;
                    prev_f = curr_f;
                }
            }

            if (!found) {
                // Extend bracket range further (like reference Python: p_min *= 0.01, p_max *= 100)
                P_lo = P_min * Tscal{1e-8};
                P_hi = P_max * Tscal{1e8};
                f_lo = f(P_lo);
                f_hi = f(P_hi);

                // Final attempt: scan with finer resolution
                if (f_lo * f_hi > Tscal{0} || !is_valid(f_lo) || !is_valid(f_hi)) {
                    const Tscal log_P_lo_ext = sycl::log10(P_lo);
                    const Tscal log_P_hi_ext = sycl::log10(P_hi);

                    prev_P = P_lo;
                    prev_f = f_lo;

                    for (int i = 1; i <= 50 && !found; ++i) {
                        const Tscal log_P
                            = log_P_lo_ext + (log_P_hi_ext - log_P_lo_ext) * Tscal(i) / Tscal{50};
                        const Tscal curr_P = sycl::pow(Tscal{10}, log_P);
                        const Tscal curr_f = f(curr_P);

                        // Only use valid values for bracketing (like reference)
                        if (is_valid(curr_f) && is_valid(prev_f) && prev_f * curr_f < Tscal{0}) {
                            P_lo  = prev_P;
                            P_hi  = curr_P;
                            f_lo  = prev_f;
                            f_hi  = curr_f;
                            found = true;
                        }
                        if (is_valid(curr_f)) {
                            prev_P = curr_P;
                            prev_f = curr_f;
                        }
                    }
                }

                // If still not found, use simple average pressure (NO acoustic fallback)
                if (!found) {
                    result.P_star    = Tscal{0.5} * (P_L + P_R);
                    result.v_x_star  = Tscal{0.5} * (v_x_L + v_x_R);
                    result.v_t_star  = (result.v_x_star >= Tscal{0}) ? v_t_L : v_t_R;
                    result.converged = false; // Mark as NOT converged
                    return result;
                }
            }
        }

        // Hybrid Newton-bisection
        Tscal P_star = sycl::sqrt(P_lo * P_hi); // Geometric mean initial guess

        for (u32 iter = 0; iter < max_iter; ++iter) {
            const Tscal f_mid = f(P_star);

            // Check for NaN (like reference)
            if (!is_valid(f_mid)) {
                // Return fallback on NaN
                result.P_star    = Tscal{0.5} * (P_L + P_R);
                result.v_x_star  = Tscal{0.5} * (v_x_L + v_x_R);
                result.v_t_star  = Tscal{0};
                result.converged = false;
                return result;
            }

            // Check convergence
            if (sycl::fabs(f_mid) < tol || sycl::fabs(P_hi - P_lo) < tol * P_star) {
                result.converged = true;
                break;
            }

            // Update bracket
            if (f_mid * f_lo < Tscal{0}) {
                P_hi = P_star;
                f_hi = f_mid;
            } else {
                P_lo = P_star;
                f_lo = f_mid;
            }

            // Numerical derivative
            const Tscal dP      = P_star * Tscal{1e-6};
            const Tscal f_plus  = f(P_star + dP);
            const Tscal f_minus = f(P_star - dP);
            const Tscal df      = (f_plus - f_minus) / (Tscal{2} * dP);

            // Newton step
            bool use_newton = false;
            Tscal P_newton  = P_star;
            if (is_valid(df) && sycl::fabs(df) > Tscal{1e-20}) {
                P_newton = P_star - f_mid / df;
                // Accept Newton step only if it stays strictly within bracket (like reference)
                if (P_newton > P_lo && P_newton < P_hi) {
                    use_newton = true;
                }
            }

            if (use_newton) {
                P_star = P_newton;
            } else {
                // Bisection (geometric mean)
                P_star = sycl::sqrt(P_lo * P_hi);
            }
        }

        result.P_star = P_star;

        // FAIL-FAST: Check P_star is valid before proceeding
        if (!sycl::isfinite(P_star) || P_star <= Tscal{0}) {
            // Return average values for debugging (allows simulation to continue
            // while we investigate the root cause)
            result.P_star    = Tscal{0.5} * (P_L + P_R);
            result.v_x_star  = Tscal{0.5} * (v_x_L + v_x_R);
            result.v_t_star  = Tscal{0};
            result.converged = false;
            return result;
        }

        // Compute star state velocities
        Tscal v_x_L_star, n_L_star, H_L_star, v_t_L_star;
        Tscal v_x_R_star, n_R_star, H_R_star, v_t_R_star;

        if (P_star > P_L) {
            solve_shock(
                P_star,
                P_L,
                n_L,
                v_x_L,
                v_t_L,
                gamma,
                true,
                v_x_L_star,
                n_L_star,
                H_L_star,
                v_t_L_star);
        } else {
            solve_rarefaction(
                P_star,
                P_L,
                n_L,
                v_x_L,
                v_t_L,
                gamma,
                true,
                v_x_L_star,
                n_L_star,
                H_L_star,
                v_t_L_star);
        }

        if (P_star > P_R) {
            solve_shock(
                P_star,
                P_R,
                n_R,
                v_x_R,
                v_t_R,
                gamma,
                false,
                v_x_R_star,
                n_R_star,
                H_R_star,
                v_t_R_star);
        } else {
            solve_rarefaction(
                P_star,
                P_R,
                n_R,
                v_x_R,
                v_t_R,
                gamma,
                false,
                v_x_R_star,
                n_R_star,
                H_R_star,
                v_t_R_star);
        }

        // Average normal velocity (should match at convergence)
        result.v_x_star = Tscal{0.5} * (v_x_L_star + v_x_R_star);

        // Tangent velocity: upwind based on contact direction
        if (result.v_x_star > Tscal{0}) {
            result.v_t_star = v_t_L_star;
        } else {
            result.v_t_star = v_t_R_star;
        }

        // Ensure subluminal
        const Tscal v_x_star2 = result.v_x_star * result.v_x_star;
        const Tscal v_t_max   = sycl::sqrt(sycl::fmax(Tscal{0}, Tscal{0.9999} - v_x_star2));
        if (sycl::fabs(result.v_t_star) > v_t_max) {
            result.v_t_star = sycl::copysign(v_t_max * Tscal{0.99}, result.v_t_star);
        }

        // FAIL-FAST: Final validation - any NaN/Inf means solver failed
        if (!sycl::isfinite(result.P_star) || !sycl::isfinite(result.v_x_star)
            || !sycl::isfinite(result.v_t_star)) {
            result.P_star    = Tscal{0.5} * (P_L + P_R);
            result.v_x_star  = Tscal{0.5} * (v_x_L + v_x_R);
            result.v_t_star  = Tscal{0};
            result.converged = false;
        }

        return result;
    }

} // namespace shammodels::gsph::physics::sr::riemann
