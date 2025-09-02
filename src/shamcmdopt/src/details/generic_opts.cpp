// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/details/generic_opts.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/tty.hpp"
#include <string_view>
#include <vector>

/**
 * @brief List of known terminal ident that support colors
 */
static const std::vector<std::string_view> color_suport_term{
    "xterm",
    "xterm-256",
    "xterm-256color",
    "xterm-truecolor",
    "vt100",
    "color",
    "ansi",
    "cygwin",
    "linux",
    "xterm-kitty",
    "alacritty"};

/**
 * @brief detect if terminal emulator support colored outputs
 *
 * @return true
 * @return false
 */
bool term_support_color() {

    auto term_var = shamcmdopt::getenv_str("TERM");
    if (term_var) {
        for (auto term : color_suport_term) {
            if (*term_var == term) {
                return true;
            }
        }
    }

    auto colorterm_var = shamcmdopt::getenv_str("COLORTERM");
    if (colorterm_var) {
        if (*colorterm_var == "truecolor") {
            return true;
        }
        if (*colorterm_var == "24bit") {
            return true;
        }
    }

    return false;
}

namespace shamcmdopt {

    void register_cmdopt_generic_opts() {

        register_opt("--nocolor", {}, "disable colored ouput");
        register_opt("--color", {}, "force colored ouput");
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

        bool has_envvar_nocolor = bool(getenv_str("NO_COLOR"));
        bool has_envvar_color   = bool(getenv_str("CLICOLOR_FORCE"));

        if (has_opt_color && has_opt_nocolor) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "You can not pass --nocolor and --color simultaneously");
        }

        if (has_opt_nocolor) {
            shambase::term_colors::disable_colors();
        } else if (has_opt_color) {
            shambase::term_colors::enable_colors();
        } else if (has_envvar_nocolor) {
            shambase::term_colors::disable_colors();
        } else if (has_envvar_color) {
            shambase::term_colors::enable_colors();
        } else {
            if (term_support_color() && is_a_tty()) {
                shambase::term_colors::enable_colors();
            } else {
                shambase::term_colors::disable_colors();
            }
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
        auto res = getenv_str("SHAMTTYCOL");

        int min_sz = 10;
        if (res) {
            try {
                try {
                    int val = std::stoi(*res);
                    if (val < min_sz) {
                        val = min_sz;
                    }
                    set_tty_columns(val);
                } catch (const std::invalid_argument &a) {
                    shambase::println("Error : SHAMTTYCOL is not an integer");
                }
            } catch (const std::out_of_range &a) {
                shambase::println("Error : SHAMTTYCOL is out of range");
            }
        }
    }

    void process_cmdopt_generic_opts() {

        process_colors();
        process_tty();

        if (has_option("--help")) {
            print_help();

            shambase::println("\nEnv deduced vars :");

            if (is_a_tty()) {
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
                shambase::format("  tty size = {}x{}", get_tty_lines(), get_tty_columns()));
        }
    }

} // namespace shamcmdopt
