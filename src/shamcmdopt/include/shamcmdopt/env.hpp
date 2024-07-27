// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file env.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include <optional>
#include <string>

namespace shamcmdopt {

    /**
     * @brief Get the content of the environment variable if it exist
     * 
     * @param env_var the name of the env variable
     * @return std::optional<std::string> return the value of the env variable if it exist, none otherwise
     */
    std::optional<std::string> getenv_str(const char *env_var);

    void register_env_var_doc(std::string env_var, std::string desc);

    void print_help_env_var();

}
