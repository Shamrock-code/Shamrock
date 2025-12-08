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

    template<class T>
    struct EOS_Isothermal {

        static constexpr T pressure(T cs, T rho) { return cs * cs * rho; }
    };

    template<class T>
    struct EOS_Adiabatic {

        static constexpr T pressure(T gamma, T rho, T u) { return (gamma - 1) * rho * u; }

        static constexpr T soundspeed(T gamma, T rho, T u) {
            return sycl::sqrt(gamma * eos_adiabatic(gamma, rho, u) / rho);
        }

        static constexpr T cs_from_p(T gamma, T rho, T P) { return sycl::sqrt(gamma * P / rho); }
    };

    template<class T>
    struct EOS_Polytropic {

        static constexpr T pressure(T gamma, T K, T rho) { return K * sycl::pow(rho, gamma); }

        static constexpr T soundspeed(T gamma, T K, T rho) {
            return sycl::sqrt(gamma * pressure(gamma, K, rho) / rho);
        }

        static constexpr T polytropic_index(T n) { return 1. + 1. / n; }
    };

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
    struct EOS_Fermi {
        static constexpr T pi  = shamunits::pi<T>;
        static constexpr T h   = shamunits::Constants<T>::Si::h;
        static constexpr T m_e = shamunits::Constants<T>::Si::electron_mass;
        static constexpr T m_p = shamunits::Constants<T>::Si::proton_mass;
        static constexpr T c   = shamunits::Constants<T>::Si::c;
        static constexpr T coeff_p
            = pi * m_e * m_e * m_e * m_e * c * c * c * c * c / (3 * h * h * h);
        static constexpr T coeff_pf = 3 * h * h * h / (8 * pi * m_p);

        //= \tilde p_F = Fermi momentum divided by m_e*c
        static constexpr T tpf(T mu_e, T rho) {
            return sycl::rootn(coeff_pf * rho / mu_e, 3) / (m_e * c);
        }

        struct PressureAndCs {
            T pressure;
            T soundspeed;
        };
        static constexpr PressureAndCs pressure_and_soundspeed(T mu_e, T rho) {
            T pf  = tpf(mu_e, rho);
            T pf2 = pf * pf;
            T P   = coeff_p * (pf * sycl::sqrt(pf2 + 1) * (2 * pf2 - 3) + 3 * sycl::asinh(pf));
            T cs2 = 8 * coeff_pf * coeff_p * pf2 * pf2
                    / (3 * mu_e * sycl::powr(rho, 2. / 3.) * sycl::sqrt(1 + pf2));
            return {P, sycl::sqrt(cs2)};
        }
    };

} // namespace shamphys
