// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SolverLog.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shammodels/sph/SolverLog.hpp"
#include <stdexcept>

f64 shammodels::sph::SolverLog::get_last_rate() {

    if (step_logs.size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size() - 1;

    u64 obj_loc  = step_logs[last_id].rank_count;
    f64 time_loc = step_logs[last_id].elasped_sec;

    u64 obj_total = shamalgs::collective::allreduce_sum(obj_loc);
    f64 max_t     = shamalgs::collective::allreduce_max(time_loc);

    return f64(obj_total) / max_t;
}

u64 shammodels::sph::SolverLog::get_last_obj_count() {

    if (step_logs.size() == 0) {
        throw shambase::make_except_with_loc<std::runtime_error>("");
    }

    u64 last_id = step_logs.size() - 1;

    u64 cnt_loc = step_logs[last_id].rank_count;

    u64 cnt_tot = shamalgs::collective::allreduce_sum(cnt_loc);

    return cnt_tot;
}
