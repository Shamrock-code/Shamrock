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
 * @file SolverLog.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <optional>
#include <vector>

namespace shammodels::sph {
    struct SolverLog;
} // namespace shammodels::sph

/**
 * @brief Class holding the logs of the solver
 * /todo add a variable to keep only a definite number of steps in the step_logs
 */
struct shammodels::sph::SolverLog {

    struct StepInfo {
        f64 solver_t;
        f64 solver_dt;
        i32 world_rank;
        u64 rank_count;
        f64 rate;
        f64 elasped_sec;
        f64 wtime;
    };

    std::vector<StepInfo> step_logs;

    inline void register_log(StepInfo info) { step_logs.push_back(info); }

    f64 get_last_rate();
    u64 get_last_obj_count();

    u64 get_iteration_count() { return step_logs.size(); }
};
