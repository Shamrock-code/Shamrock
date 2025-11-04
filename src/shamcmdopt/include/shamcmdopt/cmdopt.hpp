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
 * @file cmdopt.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <string_view>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace shamcmdopt {

    /**
     * @brief Get the number of command line arguments.
     *
     * @return int The number of command line arguments.
     */
    int get_argc();

    /**
     * @brief Get the command line arguments.
     *
     * @return char** The command line arguments.
     */
    char **get_argv();

    /**
     * @brief Register a command line option.
     *
     * @param name The name of the option.
     * @param args The arguments of the option. If the option takes no argument,
     *             pass std::nullopt.
     * @param description The description of the option.
     *
     */
    void register_opt(std::string name, std::optional<std::string> args, std::string description);

    /**
     * @brief Initialize the command line option parser.
     * We also process generic options in this call (color detection typically)
     *
     * @param argc The number of command line arguments.
     * @param argv The command line arguments.
     */
    void init(int argc, char *argv[]);

    /**
     * @brief Check if an option is present.
     *
     * @param option_name The name of the option.
     * @return true If the option is present.
     * @return false If the option is not present.
     */
    bool has_option(const std::string_view &option_name);

    /**
     * @brief Get the value of an option.
     *
     * @param option_name The name of the option.
     * @return std::string_view The value of the option.
     */
    std::string_view get_option(const std::string_view &option_name);

    /**
     * @brief Print the help message.
     */
    void print_help();

    /**
     * @brief Check if the help mode is enabled.
     *
     * @return true If the help mode is enabled.
     * @return false If the help mode is not enabled.
     */
    bool is_help_mode();

} // namespace shamcmdopt

namespace opts {
    using namespace shamcmdopt;
}
