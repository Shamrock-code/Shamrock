// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
