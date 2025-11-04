// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HydroSoundwave.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamphys/HydroSoundwave.hpp"

auto shamphys::HydroSoundwave::get_value(f64 t, f64 x) -> field_val {
    static constexpr std::complex<double> i(0.0, 1.0);

    std::complex<f64> val = std::exp(i * (get_omega() * t - k * x));

    return {std::real(val * rho_tilde), std::real(val * v_tilde)};
}
