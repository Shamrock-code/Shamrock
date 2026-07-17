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
 * @file SynchronizedData.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Scheduler data guaranteed to be in sync across all ranks.
 */

#include "shamrock/solvergraph/SolverGraphSerializable.hpp"
#include <nlohmann/json_fwd.hpp>

/// Data stored within the scheduler that are garanteed to be in sink across all ranks
struct SynchronizedData {
    shamrock::solvergraph::SolverGraphSerializable container = {};

    nlohmann::json to_json();

    void from_json(const nlohmann::json &j);
};
