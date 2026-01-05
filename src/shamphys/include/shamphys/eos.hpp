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
 * @file eos.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

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

        static constexpr T temperature(T P, T rho, T mu, T mh, T kb) {
            return mu * mh * P / (rho * kb);
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
        };
    };

    template<class T>
    struct PressureAndCs {
        T pressure;
        T soundspeed;
    };

    /**
     * @brief Fermi Gas EoS
     *
     * mu_e is the mean molecular weight
     *
     * Sound speed:
     * \f[
     * c_s =
     * \sqrt{\frac{8\alpha\beta}{3\mu_e^{\frac13}\rho^{\frac23}}\frac{\tilde{p}_F^4}{\sqrt{1+\tilde{p}_F^2}}}
     * \f] where:
     * \f[
     * \begin{cases}
     * \tilde{p}_F &= \frac{1}{\mu_e^{\frac13}}\alpha \rho^{1/3} \\
     * \alpha &= \frac{1}{\me c}h\left(\frac{3}{8\pi m_{\mathrm{p}}}\right)^{\frac13} \\
     * \beta &= \frac{\pi m_\mathrm{e}^4c^5}{3h^3}
     * \end{cases}
     * \f]
     *
     * Pressure:
     * \f[
     * P =  \beta
     * \left(\left[\tilde{p}_F\sqrt{\tilde{p}_F^2+1}(2\tilde{p}_F^2-3)+3\mathop{arcsinh}(\tilde{p}_F)\right]
     * \f]
     */
    template<class T>
    struct EOS_Fermi {

        static constexpr PressureAndCs<T> pressure_and_soundspeed(T mu_e, T rho) {
            // rho has to be SI !

            constexpr T ALPHA
                = 0.10064082802851738e-2; // = (3/(8pi))**(1./3) * h / (mp^(1/3) m_e c) SI
            constexpr T BETA = 6002.332181706928e18; // = (pi/3) * m_e^4c^5/h^3 SI

            //\tilde p_F = Fermi momentum divided by m_e*c
            const T mu13 = sycl::rootn(mu_e, 3);
            T tpf        = ALPHA * sycl::rootn(rho, 3) / mu13;
            T tpf2       = tpf * tpf;

            T P   = BETA * (tpf * sycl::sqrt(tpf2 + 1) * (2 * tpf2 - 3) + 3 * sycl::asinh(tpf));
            T cs2 = 8 * ALPHA * BETA * tpf2 * tpf2
                    / (3 * mu13 * sycl::powr(rho, 2. / 3.) * sycl::sqrt(1 + tpf2));
            return {P, sycl::sqrt(cs2)};
        }
    };

} // namespace shamphys
