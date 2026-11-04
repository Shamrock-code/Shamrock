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
 * @file PatchSchedulerConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Load-split/merge thresholds for PatchScheduler
 *
 * Kept in a dedicated header so solver configs can include this type without
 * the full scheduler (json, trees, MPI, SYCL).
 */

#include "shambase/aliases_int.hpp"
#include "nlohmann/json_fwd.hpp"

struct PatchSchedulerConfig {
    u64 split_load_value = 0_u64;
    u64 merge_load_value = 0_u64;
};

/**
 * @brief Converts a PatchSchedulerConfig object to a JSON object.
 *
 * @param j The JSON object to be populated.
 * @param p The PatchSchedulerConfig object to be converted.
 */
void to_json(nlohmann::json &j, const PatchSchedulerConfig &p);

/**
 * @brief Deserializes a PatchSchedulerConfig object from a JSON object.
 *
 * @param j The JSON object to deserialize from.
 * @param p The PatchSchedulerConfig object to populate.
 */
void from_json(const nlohmann::json &j, PatchSchedulerConfig &p);
