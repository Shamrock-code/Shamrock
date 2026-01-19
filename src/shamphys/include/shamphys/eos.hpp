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
 * @file eos.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"

namespace shamphys {

    /**
     * @brief Isothermal equation of state
     *
     * Pressure: \f$ P = c_s^2 \rho \f$
     */
    template<class T>
    struct EOS_Isothermal {

        static constexpr T pressure(T cs, T rho) { return cs * cs * rho; }
    };

    /**
     * @brief Adiabatic equation of state
     *
     * Pressure: \f$ P = (\gamma - 1) \rho u \f$
     *
     * Sound speed: \f$ c_s = \sqrt{\frac{\gamma P}{\rho}} \f$
     */
    template<class T>
    struct EOS_Adiabatic {

        static constexpr T pressure(T gamma, T rho, T u) { return (gamma - 1) * rho * u; }

        static constexpr T soundspeed(T gamma, T rho, T u) {
            return sycl::sqrt(gamma * eos_adiabatic(gamma, rho, u) / rho);
        }

        static constexpr T cs_from_p(T gamma, T rho, T P) { return sycl::sqrt(gamma * P / rho); }
    };

    /**
     * @brief Polytropic equation of state
     *
     * Pressure: \f$ P = K \rho^\gamma \f$
     *
     * Sound speed: \f$ c_s = \sqrt{\frac{\gamma P}{\rho}} = \sqrt{\gamma K \rho^{\gamma-1}} \f$
     *
     * Polytropic index: \f$ \gamma = 1 + \frac{1}{n} \f$
     */
    template<class T>
    struct EOS_Polytropic {

        static constexpr T pressure(T gamma, T K, T rho) { return K * sycl::pow(rho, gamma); }

        static constexpr T soundspeed(T gamma, T K, T rho) {
            return sycl::sqrt(gamma * pressure(gamma, K, rho) / rho);
        }

        static constexpr T polytropic_index(T n) { return 1. + 1. / n; }
    };

    /**
     * @brief Locally isothermal equation of state with radial dependence
     *
     * Sound speed squared: \f$ c_s^2(R) = c_{s,0}^2 R^{2q} \f$
     *
     * Pressure: \f$ P = c_s^2(R) \rho = c_{s,0}^2 R^{2q} \rho \f$
     *
     * where \f$ R \f$ is the radial distance and \f$ q \f$ is the power-law index.
     */
    template<class T>
    struct EOS_LocallyIsothermal {

        static constexpr T soundspeed_sq(T cs0sq, T Rsq, T mq) {
            return cs0sq * sycl::pow(Rsq, mq);
        }

        static constexpr T pressure(T cs0sq, T Rsq, T mq, T rho) {
            return soundspeed_sq(cs0sq, Rsq, mq) * rho;
        }

        static constexpr T pressure_from_cs(T cs0sq, T rho) { return cs0sq * rho; }
    };

    /**
     * @brief Piecewise polytropic EOS from Machida et al. (2006)
     *
     * Uses different gamma values across density thresholds for gravitational collapse modeling.
     *
     * Sound speed: \f$ c_s = \sqrt{\frac{\gamma P}{\rho}} \f$ where \f$ \gamma \f$ depends on
     * density:
     * \f[
     * \gamma = \begin{cases}
     *   1.0 & \rho < \rho_{c1} \\
     *   7/5 & \rho_{c1} \leq \rho < \rho_{c2} \\
     *   1.1 & \rho_{c2} \leq \rho < \rho_{c3} \\
     *   5/3 & \rho \geq \rho_{c3}
     * \end{cases}
     * \f]
     *
     * Pressure (piecewise):
     * \f[
     * P = \begin{cases}
     *   c_s^2 \rho & \rho < \rho_{c1} \\
     *   c_s^2 \rho_{c1} \left(\frac{\rho}{\rho_{c1}}\right)^{7/5} & \rho_{c1} \leq \rho < \rho_{c2}
     * \\
     *   c_s^2 \rho_{c1} \left(\frac{\rho_{c2}}{\rho_{c1}}\right)^{7/5}
     * \left(\frac{\rho}{\rho_{c2}}\right)^{1.1} & \rho_{c2} \leq \rho < \rho_{c3} \\ c_s^2
     * \rho_{c1} \left(\frac{\rho_{c2}}{\rho_{c1}}\right)^{7/5}
     * \left(\frac{\rho_{c3}}{\rho_{c2}}\right)^{1.1} \left(\frac{\rho}{\rho_{c3}}\right)^{5/3} &
     * \rho \geq \rho_{c3}
     * \end{cases}
     * \f]
     *
     * Temperature: \f$ T = \frac{\mu m_H P}{\rho k_B} \f$
     */
    template<class T>
    struct EOS_Machida06 {

        static constexpr T soundspeed(T P, T rho, T rho_c1, T rho_c2, T rho_c3) {
            const T gamma = (rho < rho_c1)   ? T(1.0)
                            : (rho < rho_c2) ? T(7.0 / 5.0)
                            : (rho < rho_c3) ? T(1.1)
                                             : T(5.0 / 3.0);
            return sycl::sqrt(gamma * P / rho);
        }

        static constexpr T temperature(T P, T rho, T chi, T mh, T kb) {
            return chi * mh * P / (rho * kb);
        }

        static constexpr T pressure(T cs, T rho, T rho_c1, T rho_c2, T rho_c3) {
            if (rho < rho_c1) {
                return cs * cs * rho;
            } else if (rho < rho_c2) {
                return cs * cs * rho_c1 * sycl::pow(rho / rho_c1, 7. / 5.);
            } else if (rho < rho_c3) {
                return cs * cs * rho_c1 * sycl::pow(rho_c2 / rho_c1, 7. / 5.)
                       * sycl::pow(rho / rho_c2, 1.1);
            } else {
                return cs * cs * rho_c1 * sycl::pow(rho_c2 / rho_c1, 7. / 5.)
                       * sycl::pow(rho_c3 / rho_c2, 1.1) * sycl::pow(rho / rho_c3, 5. / 3.);
            }
        }
    };

    template<class T>
    struct PressureAndCs {
        T pressure;
        T soundspeed;
    };

    template<class T>
    struct EOS_Tillotson {

        static PressureAndCs<T> pressure_and_cs(
            T rho, T u, T rho0, T E0, T A, T B, T a, T b, T alpha, T beta, T u_iv, T u_cv) {

            T eta    = rho / rho0;
            T eta2   = eta * eta;
            T chi    = eta - 1.0;
            T omega  = u / (E0 * eta2);
            T denom  = 1.0 + omega;
            T denom2 = denom * denom;

            T P       = 0.0;
            T dP_drho = 0.0;
            T dP_du   = 0.0;

            // --- 1. Condensed/Cold state (rho > rho0 || u < u_iv) ---
            auto compute_cold = [&]() {
                // P_c formula
                T term_bracket = a + b / denom;
                T P_c          = term_bracket * rho * u + A * chi + B * chi * chi;

                if (P_c < 0.0) {
                    P_c = 0.0;
                }

                T term_rho_1 = u * term_bracket;
                T term_rho_2 = (u * u / E0) * (2.0 / eta2) * (b / denom2);
                T term_rho_3 = (1.0 / rho0) * (A + 2.0 * B * chi);

                T dPc_drho = term_rho_1 + term_rho_2 + term_rho_3;
                T dPc_du   = rho * (a + b / denom2);

                return std::make_tuple(P_c, dPc_drho, dPc_du);
            };

            // --- 2. Expanded, vaporized (rho < rho0 && u > u_cv) ---
            auto compute_hot = [&]() {
                T X = (rho0 / rho) - 1.0;

                T exp_beta  = sycl::exp(-beta * X);
                T exp_alpha = sycl::exp(-alpha * X * X);

                // P_h formula
                // P = a*rho*u + [ b*rho*u / (1+w) + A*chi*e^(-beta*X) ] * e^(-alpha*X^2)
                T part_a = a * rho * u;
                T part_b = (b * rho * u) / denom;
                T part_A = A * chi * exp_beta;

                T P_h = part_a + (part_b + part_A) * exp_alpha;

                // dPh / du
                // rho * [ a + (b / (1+w)^2) * exp_alpha ]
                T dPh_du = rho * (a + (b / denom2) * exp_alpha);

                // dPh / drho
                T dPartA_drho = a * u;
                T term_brackets_b
                    = (1.0 / rho) * (1.0 + (2.0 * omega / denom) + (2.0 * alpha * X / eta));
                T dPartB_drho      = (part_b * exp_alpha) * term_brackets_b;
                T brackets_A       = 1.0 + (chi / eta2) * (beta + 2.0 * alpha * X);
                T dPartA_term_drho = (A / rho0) * exp_beta * exp_alpha * brackets_A;

                T dPh_drho = dPartA_drho + dPartB_drho + dPartA_term_drho;

                return std::make_tuple(P_h, dPh_drho, dPh_du);
            };

            if (rho >= rho0 || u < u_iv) {
                auto [p, dpdrho, dpdu] = compute_cold();
                P                      = p;
                dP_drho                = dpdrho;
                dP_du                  = dpdu;
            } else if (u > u_cv) {
                auto [p, dpdrho, dpdu] = compute_hot();
                P                      = p;
                dP_drho                = dpdrho;
                dP_du                  = dpdu;
            } else {
                auto [Pc, dPc_drho, dPc_du] = compute_cold();
                auto [Ph, dPh_drho, dPh_du] = compute_hot();

                T delta_U = u_cv - u_iv;
                T x       = (u - u_iv) / delta_U;

                // P = (1-x)Pc + xPh
                P = (1.0 - x) * Pc + x * Ph;

                // dP/drho
                dP_drho = (1.0 - x) * dPc_drho + x * dPh_drho;

                // dP/du
                // P' = x' * (Ph - Pc) + (1-x)Pc' + xPh'
                // x' = 1 / delta_U
                T term_interp = (Ph - Pc) / delta_U;
                T term_derivs = (1.0 - x) * dPc_du + x * dPh_du;
                dP_du         = term_interp + term_derivs;
            }

            T c2 = dP_drho + (P / (rho * rho)) * dP_du;
            if (c2 < 0.0) {
                c2 = 0.0;
            }

            return {P, sycl::sqrt(c2)};
        }
    };

} // namespace shamphys
