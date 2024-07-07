// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file HydroSoundwave.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamphys/HydroSoundwave.hpp"

auto shamphys::HydroSoundwave::get_value(f64 t, f64 x) -> field_val {
    static constexpr std::complex<double> i(0.0, 1.0);

    std::complex<f64> val = std::exp(i * (get_omega() * t - k * x));

    return {std::real(val * rho_tilde), std::real(val * v_tilde)};
}
