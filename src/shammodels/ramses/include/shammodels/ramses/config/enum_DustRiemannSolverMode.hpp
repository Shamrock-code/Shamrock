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
 * @file enum_DustRiemannSolverMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"

namespace shammodels::basegodunov {

    enum DustRiemannSolverMode {
        NoDust = 0,
        DHLL   = 1, // Dust HLL . This is merely the HLL solver for dust. It's then a Rusanov like
        HB     = 2 // Huang and Bai. Pressureless Riemann solver by Huang and Bai (2022) in Athena++
    };

    inline void to_json(nlohmann::json &j, const DustRiemannSolverMode &p) {
        switch (p) {
        case DustRiemannSolverMode::NoDust: j = "no_dust"; return;
        case DustRiemannSolverMode::DHLL  : j = "dhll"; return;
        case DustRiemannSolverMode::HB    : j = "hb"; return;
        }
        throw shambase::make_except_with_loc<std::runtime_error>(
            "Invalid dust Riemann solver mode: " + std::to_string(p));
    }

    inline void from_json(const nlohmann::json &j, DustRiemannSolverMode &p) {
        std::string dust_solver;
        j.get_to(dust_solver);
        if (dust_solver == "no_dust") {
            p = DustRiemannSolverMode::NoDust;
        } else if (dust_solver == "dhll") {
            p = DustRiemannSolverMode::DHLL;
        } else if (dust_solver == "hb") {
            p = DustRiemannSolverMode::HB;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid dust Riemann solver mode: " + dust_solver);
        }
    }

} // namespace shammodels::basegodunov
