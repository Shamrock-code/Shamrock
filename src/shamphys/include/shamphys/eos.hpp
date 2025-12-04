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

} // namespace shamphys
