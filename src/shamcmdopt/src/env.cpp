// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file env.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shamcmdopt/env.hpp"

std::optional<std::string> shamcmdopt::getenv_str(const char *env_var) {
    const char *val = std::getenv(env_var);
    if (val != nullptr) {
        return std::string(val);
    }
    return {};
}
