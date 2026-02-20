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
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"

namespace shammodels::basegodunov {

    enum SlopeMode {
        None        = 0,
        VanLeer_f   = 1,
        VanLeer_std = 2,
        VanLeer_sym = 3,
        Minmod      = 4,
    };

    inline void to_json(nlohmann::json &j, const SlopeMode &p) {
        switch (p) {
        case SlopeMode::None       : j = "none"; return;
        case SlopeMode::VanLeer_f  : j = "vanleer_f"; return;
        case SlopeMode::VanLeer_std: j = "vanleer_std"; return;
        case SlopeMode::VanLeer_sym: j = "vanleer_sym"; return;
        case SlopeMode::Minmod     : j = "minmod"; return;
        }
        throw shambase::make_except_with_loc<std::runtime_error>(
            "Invalid slope mode: " + std::to_string(p));
    }

    inline void from_json(const nlohmann::json &j, SlopeMode &p) {
        std::string slope_mode;
        j.at("slope_mode").get_to(slope_mode);
        if (slope_mode == "none") {
            p = SlopeMode::None;
        } else if (slope_mode == "vanleer_f") {
            p = SlopeMode::VanLeer_f;
        } else if (slope_mode == "vanleer_std") {
            p = SlopeMode::VanLeer_std;
        } else if (slope_mode == "vanleer_sym") {
            p = SlopeMode::VanLeer_sym;
        } else if (slope_mode == "minmod") {
            p = SlopeMode::Minmod;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid slope mode: " + slope_mode);
        }
    }

} // namespace shammodels::basegodunov
