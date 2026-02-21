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
 * @file enum_SlopeMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Slope mode enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    /// Slope limiter modes
    enum SlopeMode {
        None        = 0, //< slope is zero (fallback to order 1 reconstruction)
        VanLeer_f   = 1, //< Van Leer flux limiter (Toro form, see slope_function_van_leer_f_form)
        VanLeer_std = 2, //< Van Leer standard flux limiter (see shammath::van_leer_slope)
        VanLeer_sym = 3, //< Van Leer symmetric flux limiter (see shammath::van_leer_slope_symetric)
        Minmod      = 4, //< Minmod flux limiter (see shammath::minmod)
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        SlopeMode,
        {{SlopeMode::None, "none"},
         {SlopeMode::VanLeer_f, "vanleer_f"},
         {SlopeMode::VanLeer_std, "vanleer_std"},
         {SlopeMode::VanLeer_sym, "vanleer_sym"},
         {SlopeMode::Minmod, "minmod"}});

} // namespace shammodels::basegodunov
