// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file generic_opts.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file handler generic cli & env options
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "sham/term/color_env.hpp"
#include "sham/term/tty.hpp"
#include "sham/term/tty_env.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/details/generic_opts.hpp"
#include "shamcmdopt/env.hpp"
#include <string_view>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    auto term_parse_error_callback(const char *what, std::source_location where)
        -> std::invalid_argument {
        return shambase::make_except_with_loc<std::invalid_argument>(what, SourceLocation{where});
    };

} // namespace

namespace shamcmdopt {

    void register_cmdopt_generic_opts() {

        register_opt("--nocolor", {}, "disable colored output");
        register_opt("--color", {}, "force colored output");
        register_opt("--help", {}, "show this message");

        register_env_var_doc("NO_COLOR", "Disable colors (if no color cli args are passed)");
        register_env_var_doc("CLICOLOR_FORCE", "Enable colors (if no color cli args are passed)");
        register_env_var_doc("TERM", "Terminal emulator identifier");
        register_env_var_doc("COLORTERM", "Terminal color support identifier");
        register_env_var_doc("SHAMTTYCOL", "Set tty assumed column count");
    }

    /**
     * @brief Detect if the current process should use colored output or not
     *
     * Colors are disabled by the cli flag `--nocolor` or env variable `NO_COLOR`
     * Colors are enabled by the cli flag `--color` or env variable `CLICOLOR_FORCE`
     * If no options are set we enable colors if in tty mode and
     *   if the terminal support colored output
     *
     * - We first check cli args as they are dominant over env variables
     * - We then check if some env variables are set
     * - If nothing is set we enable color if term_support_color() and is_a_tty()
     *     returns true
     *
     */
    void process_colors() {

        bool has_opt_nocolor = has_option("--nocolor");
        bool has_opt_color   = has_option("--color");

        if (has_opt_color && has_opt_nocolor) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "You can not pass --nocolor and --color simultaneously");
        }

        auto TERM           = getenv_str_view("TERM");
        auto COLORTERM      = getenv_str_view("COLORTERM");
        auto NO_COLOR       = getenv_str_view("NO_COLOR");
        auto CLICOLOR_FORCE = getenv_str_view("CLICOLOR_FORCE");

        sham::term::parse_terminal_support(
            {
                .TERM           = TERM,
                .COLORTERM      = COLORTERM,
                .NO_COLOR       = NO_COLOR,
                .CLICOLOR_FORCE = CLICOLOR_FORCE,
            },
            term_parse_error_callback);

        if (has_opt_nocolor) {
            shambase::term_colors::disable_colors();
        } else if (has_opt_color) {
            shambase::term_colors::enable_colors();
        }
    }

    /**
     * @brief Process the SHAMTTYCOL environment variable to set the number of columns for the
     * terminal.
     *
     * If the variable is set, its value is parsed as an integer and used to set the terminal
     * columns. If the value is less than the minimum size (10), it is set to the minimum size. If
     * the value is not an integer or is out of range, an error message is printed.
     */
    void process_tty() {
        auto SHAMTTYCOL = getenv_str_view("SHAMTTYCOL");

        sham::term::parse_tty_env_vars(
            sham::term::TtyEnvVars{.SHAMTTYCOL = SHAMTTYCOL}, term_parse_error_callback);
    }

    void process_cmdopt_generic_opts() {

        process_colors();
        process_tty();

        if (has_option("--help")) {
            print_help();

            shambase::println("\nEnv deduced vars :");

            if (sham::term::is_a_tty()) {
                shambase::println("  isatty = Yes");
            } else {
                shambase::println("  isatty = No");
            }

            if (shambase::term_colors::colors_enabled()) {
                shambase::println("  color = enabled");
            } else {
                shambase::println("  color = disabled");
            }

            shambase::println(
                shambase::format(
                    "  tty size = {}x{}",
                    sham::term::get_tty_lines(),
                    sham::term::get_tty_columns()));
        }
    }

} // namespace shamcmdopt
