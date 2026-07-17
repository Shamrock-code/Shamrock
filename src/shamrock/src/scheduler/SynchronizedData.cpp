// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SynchronizedData.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief JSON serialization for scheduler synchronized data.
 */

#include "shamrock/scheduler/SynchronizedData.hpp"
#include <nlohmann/json.hpp>

nlohmann::json SynchronizedData::to_json() { return {{"solvergraph", container}}; }

void SynchronizedData::from_json(const nlohmann::json &j) { j.at("solvergraph").get_to(container); }
