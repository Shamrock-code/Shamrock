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
 * @file enum_DragSolverMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"

namespace shammodels::basegodunov {

    enum DragSolverMode {
        NoDrag = 0,
        IRK1   = 1, // Implicit RK1
        IRK2   = 2, // Implicit RK2
        EXPO   = 3  // Matrix exponential
    };

    inline void to_json(nlohmann::json &j, const DragSolverMode &p) {
        switch (p) {
        case DragSolverMode::NoDrag: j = "no_drag"; return;
        case DragSolverMode::IRK1  : j = "irk1"; return;
        case DragSolverMode::IRK2  : j = "irk2"; return;
        case DragSolverMode::EXPO  : j = "expo"; return;
        }
        throw shambase::make_except_with_loc<std::runtime_error>(
            "Invalid drag solver mode: " + std::to_string(p));
    }

    inline void from_json(const nlohmann::json &j, DragSolverMode &p) {
        std::string drag_solver;
        j.get_to(drag_solver);
        if (drag_solver == "no_drag") {
            p = DragSolverMode::NoDrag;
        } else if (drag_solver == "irk1") {
            p = DragSolverMode::IRK1;
        } else if (drag_solver == "irk2") {
            p = DragSolverMode::IRK2;
        } else if (drag_solver == "expo") {
            p = DragSolverMode::EXPO;
        } else {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "Invalid drag solver mode: " + drag_solver);
        }
    }

} // namespace shammodels::basegodunov
