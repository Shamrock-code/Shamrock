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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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
     * @brief Get the content of the environment variable if it exist, otherwise return the default
     * value
     *
     * @param env_var the name of the env variable
     * @param default_val the default value to return if the env variable does not exist
     * @return std::string the value of the env variable if it exist, the default value otherwise
     */
    inline std::string getenv_str_default(const char *env_var, std::string default_val) {
        auto val = getenv_str(env_var);
        return val ? *val : default_val;
    }

    /**
     * @brief Register the documentation of an environment variable
     *
     * This function is used to register the documentation of an environment variable.
     * The documentation will be printed in the help message.
     *
     * @throw std::invalid_argument if the environment variable is already registered
     *
     * @param env_var the name of the environment variable
     * @param desc the description of the environment variable
     */
    void register_env_var_doc(std::string env_var, std::string desc);

    /**
     * @brief Get the content of the environment variable if it exist and register it documentation
     *
     * This function is a shortcut for calling both getenv_str() and register_env_var_doc().
     * It is used to register the documentation of the environment variable and return its value if
     * it exist.
     *
     * @param env_var the name of the env variable
     * @param desc the description of the environment variable
     * @return std::optional<std::string> return the value of the env variable if it exist, none
     * otherwise
     */
    inline std::optional<std::string> getenv_str_register(const char *env_var, std::string desc) {
        register_env_var_doc(env_var, desc);
        return getenv_str(env_var);
    }

    /**
     * @brief Get the content of the environment variable if it exist and register it documentation,
     * otherwise return the default value
     *
     * This function is a shortcut for calling both getenv_str_default() and register_env_var_doc().
     * It is used to register the documentation of the environment variable and return its value if
     * it exist, otherwise return the default value.
     *
     * @param env_var the name of the env variable
     * @param default_val the default value to return if the env variable does not exist
     * @param desc the description of the environment variable
     * @return std::string the value of the env variable if it exist, the default value otherwise
     */
    inline std::string
    getenv_str_default_register(const char *env_var, std::string default_val, std::string desc) {
        register_env_var_doc(env_var, desc);
        return getenv_str_default(env_var, default_val);
    }

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
