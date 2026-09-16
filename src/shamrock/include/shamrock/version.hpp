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
 * @file version.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief typedefs and macros
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/term_colors.hpp"
#include "shamcomm/logs.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

extern const std::string git_info_str;
extern const std::string git_commit_hash;
extern const std::string compile_arg;
extern const std::string version_string;
extern const bool is_git;

// will be in compiler_id.cpp
extern const char *shamrock_compiler_id_string;

const u32 term_width = 64;
