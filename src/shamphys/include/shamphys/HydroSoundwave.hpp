// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file HydroSoundwave.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include <cmath>
#include <complex>

namespace shamphys {

    class HydroSoundwave {
        public:
        f64 cs;
        f64 k;

        std::complex<f64> rho_tilde;
        std::complex<f64> v_tilde;

        HydroSoundwave(f64 _cs, f64 _k, std::complex<f64> _rho_tilde, std::complex<f64> _v_tilde)
            : cs(_cs), k(_k), rho_tilde(_rho_tilde), v_tilde(_v_tilde) {}

        inline f64 get_omega() { return cs * k; }

        struct field_val {
            f64 rho, v;
        };

        field_val get_value(f64 t, f64 x);
    };

} // namespace shamphys
