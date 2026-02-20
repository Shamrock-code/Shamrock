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

    enum SlopeMode {
        None        = 0,
        VanLeer_f   = 1,
        VanLeer_std = 2,
        VanLeer_sym = 3,
        Minmod      = 4,
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        SlopeMode,
        {{SlopeMode::None, "none"},
         {SlopeMode::VanLeer_f, "vanleer_f"},
         {SlopeMode::VanLeer_std, "vanleer_std"},
         {SlopeMode::VanLeer_sym, "vanleer_sym"},
         {SlopeMode::Minmod, "minmod"}});

} // namespace shammodels::basegodunov
