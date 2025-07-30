// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shamcmdopt/env.hpp"

inline const std::optional<std::string> REF_FILES_PATH = shamcmdopt::getenv_str("REF_FILES_PATH");

/**
 * @brief Get the path of a reference file
 *
 * @param locpath path inside the reference file repo
 * @return std::string the absolute path
 */
inline std::string get_reffile_path(std::string locpath) {
    if (REF_FILES_PATH) {
        return *REF_FILES_PATH + "/" + locpath;
    } else {
        return std::string("reference-files") + "/" + locpath;
    }
}
