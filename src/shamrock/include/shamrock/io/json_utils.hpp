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
 * @file json_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/io/json_print_diff.hpp"

namespace shamrock {

    /// Shown the changes between two JSON objects to log config changes
    std::string log_json_changes(
        const nlohmann::json &j_current,
        const nlohmann::json &j,
        bool has_used_defaults,
        bool has_updated_config);

} // namespace shamrock
