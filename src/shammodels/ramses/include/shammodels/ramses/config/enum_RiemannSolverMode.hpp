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
 * @file enum_RiemannSolverMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"

namespace shammodels::basegodunov {

    enum RiemannSolverMode { Rusanov = 0, HLL = 1, HLLC = 2 };

    inline void to_json(nlohmann::json &j, const RiemannSolverMode &p) {
        switch (p) {
        case RiemannSolverMode::Rusanov: j = "rusanov"; return;
        case RiemannSolverMode::HLL    : j = "hll"; return;
        case RiemannSolverMode::HLLC   : j = "hllc"; return;
        }
        throw shambase::make_except_with_loc<std::runtime_error>(
            "Invalid Riemann solver mode: " + std::to_string(p));
    }

    inline void from_json(const nlohmann::json &j, RiemannSolverMode &p) {
        std::string riemann_solver;
        j.get_to(riemann_solver);
        if (riemann_solver == "rusanov") {
            p = RiemannSolverMode::Rusanov;
        } else if (riemann_solver == "hll") {
            p = RiemannSolverMode::HLL;
        } else if (riemann_solver == "hllc") {
            p = RiemannSolverMode::HLLC;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid Riemann solver mode: " + riemann_solver);
        }
    }

} // namespace shammodels::basegodunov
