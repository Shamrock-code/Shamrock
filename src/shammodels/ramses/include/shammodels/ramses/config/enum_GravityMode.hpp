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
 * @file enum_GravityMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"

namespace shammodels::basegodunov {

    enum GravityMode {
        NoGravity = 0,
        CG        = 1, // conjuguate gradient
        PCG       = 2, // preconditioned conjuguate gradient
        BICGSTAB  = 3, // bicgstab
        MULTIGRID = 4  // multigrid
    };

    inline void to_json(nlohmann::json &j, const GravityMode &p) {
        switch (p) {
        case GravityMode::NoGravity: j = "no_gravity"; return;
        case GravityMode::CG       : j = "cg"; return;
        case GravityMode::PCG      : j = "pcg"; return;
        case GravityMode::BICGSTAB : j = "bicgstab"; return;
        case GravityMode::MULTIGRID: j = "multigrid"; return;
        }
        throw shambase::make_except_with_loc<std::runtime_error>(
            "Invalid gravity mode: " + std::to_string(p));
    }

    inline void from_json(const nlohmann::json &j, GravityMode &p) {
        std::string gravity_mode;
        j.at("gravity_mode").get_to(gravity_mode);
        if (gravity_mode == "no_gravity") {
            p = GravityMode::NoGravity;
        } else if (gravity_mode == "cg") {
            p = GravityMode::CG;
        } else if (gravity_mode == "pcg") {
            p = GravityMode::PCG;
        } else if (gravity_mode == "bicgstab") {
            p = GravityMode::BICGSTAB;
        } else if (gravity_mode == "multigrid") {
            p = GravityMode::MULTIGRID;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid gravity mode: " + gravity_mode);
        }
    }

} // namespace shammodels::basegodunov
