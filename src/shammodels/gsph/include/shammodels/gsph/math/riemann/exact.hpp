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
 * @file exact.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Exact Riemann solver for GSPH (Toro 2009)
 *
 * Implements the exact solution of the 1D Riemann problem for an ideal gas:
 * the wave pattern (shock/rarefaction on each side) is classified first from
 * the initial states, then the matching closed-form shock (Rankine-Hugoniot)
 * or rarefaction (isentropic) relation is solved to convergence via bisection.
 *
 * References:
 * - Toro, E.F. (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"

namespace shammodels::gsph::riemann {

    /**
     * @brief Left/right primitive state for the exact solver (velocity along pair axis)
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct ExactState {
        Tscal v; ///< Velocity along the pair axis
        Tscal p; ///< Pressure
        Tscal r; ///< Density
    };

    /**
     * @brief Relative velocity jump across a shock wave (Rankine-Hugoniot)
     *
     * @param LR -1 for the left wave, +1 for the right wave
     * @param p2 Trial star-region pressure
     * @param p1 Pressure of the state ahead of the wave
     * @param r1 Density of the state ahead of the wave
     * @param gamma Adiabatic index
     */
    template<class Tscal>
    inline Tscal exact_v_xc_shock(Tscal LR, Tscal p2, Tscal p1, Tscal r1, Tscal gamma) {
        const Tscal inv_gp1 = Tscal{1} / (gamma + Tscal{1});
        const Tscal A       = Tscal{2} * inv_gp1 / r1;
        // Toro (2009) eq. 4.7: B_K = p_K*(gamma-1)/(gamma+1) -- pressure-dimensioned.
        const Tscal B = p1 * (gamma - Tscal{1}) * inv_gp1;
        return Tscal{-1} * LR * (p2 - p1) * sycl::sqrt(A / (p2 + B));
    }

    /**
     * @brief Relative velocity jump across a rarefaction wave (isentropic relation)
     *
     * @param LR -1 for the left wave, +1 for the right wave
     * @param p2 Trial star-region pressure
     * @param p1 Pressure of the state ahead of the wave
     * @param r1 Density of the state ahead of the wave
     * @param gamma Adiabatic index
     */
    template<class Tscal>
    inline Tscal exact_v_xc_rarefaction(Tscal LR, Tscal p2, Tscal p1, Tscal r1, Tscal gamma) {
        const Tscal cs1 = sycl::sqrt(gamma * p1 / r1);
        return Tscal{-1} * LR * Tscal{2} * cs1 / (gamma - Tscal{1})
               * (sycl::pow(p2 / p1, Tscal{0.5} * (gamma - Tscal{1}) / gamma) - Tscal{1});
    }

    template<class Tscal>
    inline Tscal exact_v_lr_ss(
        Tscal pS, ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma) {
        return exact_v_xc_shock(Tscal{-1}, pS, left.p, left.r, gamma)
               - exact_v_xc_shock(Tscal{1}, pS, right.p, right.r, gamma);
    }

    template<class Tscal>
    inline Tscal exact_v_lr_rs(
        Tscal pS, ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma) {
        return exact_v_xc_rarefaction(Tscal{-1}, pS, left.p, left.r, gamma)
               - exact_v_xc_shock(Tscal{1}, pS, right.p, right.r, gamma);
    }

    template<class Tscal>
    inline Tscal exact_v_lr_sr(
        Tscal pS, ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma) {
        return exact_v_xc_shock(Tscal{-1}, pS, left.p, left.r, gamma)
               - exact_v_xc_rarefaction(Tscal{1}, pS, right.p, right.r, gamma);
    }

    template<class Tscal>
    inline Tscal exact_v_lr_rr(
        Tscal pS, ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma) {
        return exact_v_xc_rarefaction(Tscal{-1}, pS, left.p, left.r, gamma)
               - exact_v_xc_rarefaction(Tscal{1}, pS, right.p, right.r, gamma);
    }

    /**
     * @brief Wave pattern codes: 11=shock/shock, 21=rarefaction/shock,
     * 12=shock/rarefaction, 22=rarefaction/rarefaction
     */
    template<class Tscal>
    inline i32 exact_judge_wave_pattern(
        ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma) {
        const Tscal v_lr_0 = left.v - right.v;
        if (left.p > right.p) {
            if (v_lr_0 > exact_v_lr_ss(left.p, left, right, gamma)) {
                return 11;
            } else if (v_lr_0 > exact_v_lr_rs(right.p, left, right, gamma)) {
                return 21;
            } else {
                return 22;
            }
        } else {
            if (v_lr_0 > exact_v_lr_ss(right.p, left, right, gamma)) {
                return 11;
            } else if (v_lr_0 > exact_v_lr_sr(left.p, left, right, gamma)) {
                return 12;
            } else {
                return 22;
            }
        }
    }

    /**
     * @brief Shared bisection loop for a single wave-pattern's residual function
     *
     * @param posi Initial bracket endpoint with positive residual
     * @param nega Initial bracket endpoint with non-positive residual
     * @param v_lr_0 Target value (left.v - right.v) the residual must reach
     * @param residual Wave-pattern-specific v_lr_XX(p, left, right, gamma) function
     */
    template<class Tscal, class ResidualFn>
    inline Tscal exact_bisection_generic(
        Tscal posi, Tscal nega, Tscal v_lr_0, Tscal tol, u32 max_iter, ResidualFn residual) {
        constexpr Tscal eps = Tscal{1e-16};

        Tscal half = Tscal{0};
        Tscal bis2 = residual(posi) - v_lr_0;
        for (u32 i = 0; i < max_iter; ++i) {
            const Tscal bis1 = bis2;
            half             = Tscal{0.5} * (posi + nega);
            bis2             = residual(half) - v_lr_0;

            if (sycl::fmax(sycl::fabs(bis2), sycl::fabs(bis2 - bis1) / (sycl::fabs(bis1) + eps))
                    < tol
                || bis2 == Tscal{0}) {
                break;
            }

            if (bis2 > Tscal{0}) {
                posi = half;
            } else {
                nega = half;
            }
        }
        return half;
    }

    /**
     * @brief Bisection solve for the shock/shock (11) wave pattern
     */
    template<class Tscal>
    inline Tscal exact_bisection_ss(
        ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma, Tscal tol, u32 max_iter) {
        constexpr Tscal scale_up   = Tscal{1.00001};
        constexpr Tscal scale_down = Tscal{0.99999};

        const Tscal v_lr_0 = left.v - right.v;

        // Search for an upper bound where v_lr_ss(posi) - v_lr_0 > 0. v_lr_ss(p) is
        // monotonically increasing and unbounded in p, so this always succeeds
        // within a handful of iterations for any physical input; the loop bound
        // and fallback below only guard against a degenerate/non-physical state.
        Tscal posi     = (left.p + right.p) * scale_up;
        bool bracketed = false;
        for (u32 i = 0; i < 300; ++i) {
            if (exact_v_lr_ss(posi, left, right, gamma) - v_lr_0 > Tscal{0}) {
                bracketed = true;
                break;
            }
            posi *= Tscal{10};
        }
        if (!bracketed) {
            posi = sycl::fmax(left.p, right.p);
        }
        const Tscal nega = sycl::fmin(left.p, right.p) * scale_down;

        return exact_bisection_generic(posi, nega, v_lr_0, tol, max_iter, [&](Tscal p) {
            return exact_v_lr_ss(p, left, right, gamma);
        });
    }

    /**
     * @brief Bisection solve for the rarefaction/shock (21) wave pattern
     */
    template<class Tscal>
    inline Tscal exact_bisection_rs(
        ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma, Tscal tol, u32 max_iter) {
        constexpr Tscal scale_up   = Tscal{1.00001};
        constexpr Tscal scale_down = Tscal{0.99999};

        const Tscal v_lr_0 = left.v - right.v;
        const Tscal posi   = left.p * scale_up;
        const Tscal nega   = right.p * scale_down;

        return exact_bisection_generic(posi, nega, v_lr_0, tol, max_iter, [&](Tscal p) {
            return exact_v_lr_rs(p, left, right, gamma);
        });
    }

    /**
     * @brief Bisection solve for the shock/rarefaction (12) wave pattern
     */
    template<class Tscal>
    inline Tscal exact_bisection_sr(
        ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma, Tscal tol, u32 max_iter) {
        constexpr Tscal scale_up   = Tscal{1.00001};
        constexpr Tscal scale_down = Tscal{0.99999};

        const Tscal v_lr_0 = left.v - right.v;
        const Tscal posi   = right.p * scale_up;
        const Tscal nega   = left.p * scale_down;

        return exact_bisection_generic(posi, nega, v_lr_0, tol, max_iter, [&](Tscal p) {
            return exact_v_lr_sr(p, left, right, gamma);
        });
    }

    /**
     * @brief Bisection solve for the rarefaction/rarefaction (22) wave pattern
     */
    template<class Tscal>
    inline Tscal exact_bisection_rr(
        ExactState<Tscal> left, ExactState<Tscal> right, Tscal gamma, Tscal tol, u32 max_iter) {
        constexpr Tscal scale_up = Tscal{1.00001};

        const Tscal v_lr_0 = left.v - right.v;
        const Tscal posi   = sycl::fmin(left.p, right.p) * scale_up;
        const Tscal nega   = Tscal{0};

        return exact_bisection_generic(posi, nega, v_lr_0, tol, max_iter, [&](Tscal p) {
            return exact_v_lr_rr(p, left, right, gamma);
        });
    }

    /**
     * @brief Exact Riemann solver for the 1D Euler equations (ideal gas)
     *
     * Classifies the wave pattern from the initial states, then solves the
     * matching shock/rarefaction relation to convergence via bisection.
     *
     * The left/right convention matches iterative_solver()/hllc_solver():
     * - Left state (L): particle on the "minus" side of the interface
     * - Right state (R): particle on the "plus" side of the interface
     * - Positive velocity points from L to R
     *
     * @tparam Tscal Scalar type (f32 or f64)
     * @param u_L Left state velocity (normal component)
     * @param rho_L Left state density
     * @param p_L Left state pressure
     * @param u_R Right state velocity (normal component)
     * @param rho_R Right state density
     * @param p_R Right state pressure
     * @param gamma Adiabatic index
     * @param tol Bisection convergence tolerance (default: 1e-8)
     * @param max_iter Maximum bisection iterations (default: 100)
     * @return RiemannResult with p_star and v_star
     */
    template<class Tscal>
    inline RiemannResult<Tscal> exact_solver(
        Tscal u_L,
        Tscal rho_L,
        Tscal p_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal p_R,
        Tscal gamma,
        Tscal tol    = Tscal{1.0e-8},
        u32 max_iter = 100) {

        RiemannResult<Tscal> result;

        const Tscal smallp   = Tscal{1.0e-25};
        const Tscal smallrho = Tscal{1.0e-25};

        if (rho_L < smallrho || rho_R < smallrho || p_L < smallp || p_R < smallp) {
            result.p_star = sycl::fmax(smallp, Tscal{0.5} * (p_L + p_R));
            result.v_star = Tscal{0.5} * (u_L + u_R);
            return result;
        }

        ExactState<Tscal> left{u_L, p_L, rho_L};
        ExactState<Tscal> right{u_R, p_R, rho_R};

        // Note: unlike the reference implementation this is ported from, we do
        // NOT special-case left.p == right.p here. That shortcut assumes a
        // uniform (non-evolving) state whenever pressures match, which is wrong
        // when velocities differ (e.g. the classic "123 problem" double
        // rarefaction has p_L == p_R with strongly diverging velocities). None
        // of the shock/rarefaction relations below divide by (p2 - p1), so
        // there is no numerical singularity to guard against at p_L == p_R.
        const i32 wave_pattern = exact_judge_wave_pattern(left, right, gamma);

        Tscal p_star;
        if (wave_pattern == 11) {
            p_star = exact_bisection_ss(left, right, gamma, tol, max_iter);
        } else if (wave_pattern == 21) {
            p_star = exact_bisection_rs(left, right, gamma, tol, max_iter);
        } else if (wave_pattern == 12) {
            p_star = exact_bisection_sr(left, right, gamma, tol, max_iter);
        } else {
            p_star = exact_bisection_rr(left, right, gamma, tol, max_iter);
        }
        p_star = sycl::fmax(p_star, smallp);

        const Tscal ave_v = Tscal{0.5} * (left.v + right.v);
        Tscal v_star;
        if (wave_pattern == 11) {
            v_star = ave_v
                     - Tscal{0.5}
                           * (exact_v_xc_shock(Tscal{-1}, p_star, left.p, left.r, gamma)
                              + exact_v_xc_shock(Tscal{1}, p_star, right.p, right.r, gamma));
        } else if (wave_pattern == 21) {
            v_star = ave_v
                     - Tscal{0.5}
                           * (exact_v_xc_rarefaction(Tscal{-1}, p_star, left.p, left.r, gamma)
                              + exact_v_xc_shock(Tscal{1}, p_star, right.p, right.r, gamma));
        } else if (wave_pattern == 12) {
            v_star = ave_v
                     - Tscal{0.5}
                           * (exact_v_xc_shock(Tscal{-1}, p_star, left.p, left.r, gamma)
                              + exact_v_xc_rarefaction(Tscal{1}, p_star, right.p, right.r, gamma));
        } else {
            v_star = ave_v
                     - Tscal{0.5}
                           * (exact_v_xc_rarefaction(Tscal{-1}, p_star, left.p, left.r, gamma)
                              + exact_v_xc_rarefaction(Tscal{1}, p_star, right.p, right.r, gamma));
        }

        result.p_star = p_star;
        result.v_star = v_star;
        return result;
    }

} // namespace shammodels::gsph::riemann
