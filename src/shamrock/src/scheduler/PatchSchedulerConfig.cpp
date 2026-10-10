// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchSchedulerConfig.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamrock/scheduler/PatchSchedulerConfig.hpp"
#include <nlohmann/json.hpp>

void to_json(nlohmann::json &j, const PatchSchedulerConfig &p) {
    j = nlohmann::json{
        {"split_load_value", p.split_load_value},
        {"merge_load_value", p.merge_load_value},
    };
}

void from_json(const nlohmann::json &j, PatchSchedulerConfig &p) {
    j.at("split_load_value").get_to<u64>(p.split_load_value);
    j.at("merge_load_value").get_to<u64>(p.merge_load_value);
}
