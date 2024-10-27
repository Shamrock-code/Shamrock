// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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
     * @return std::optional<std::string> return the value of the env variable if it exist, none
     * otherwise
     */
    std::optional<std::string> getenv_str(const char *env_var);

    /**
     * @brief Register the documentation of an environment variable
     *
     * This function is used to register the documentation of an environment variable.
     * The documentation will be printed in the help message.
     *
     * @param env_var the name of the environment variable
     * @param desc the description of the environment variable
     */
    void register_env_var_doc(std::string env_var, std::string desc);

    /**
     * @brief Print the documentation of the environment variables registered with
     * register_env_var_doc()
     *
     * This function is used to print the documentation of the environment variables registered with
     * register_env_var_doc(). It will print the name, the description and the value of each
     * registered environment variable.
     */
    void print_help_env_var();

} // namespace shamcmdopt
