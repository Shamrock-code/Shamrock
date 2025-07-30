// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file macroLocation.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <cstring>
#include <string>

inline std::string __file_to_loc(const char *filename) {
    return std::string(
        std::strstr(filename, "/src/") ? std::strstr(filename, "/src/") + 1 : filename);
}

inline std::string __loc_prefix(const char *filename, int line) {
    return __file_to_loc(filename) + ":" + std::to_string(line);
}

#define __FILENAME__ __file_to_loc(__FILE__)
#define __LOC_PREFIX__ __loc_prefix(__FILE__, __LINE__)

#define __LOC_POSTFIX__ ("(" + __LOC_PREFIX__ + ")")
