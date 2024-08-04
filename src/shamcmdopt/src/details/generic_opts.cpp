// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file generic_opts.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file handler generic cli & env options
 *
 */

#include "shambase/exception.hpp"
#include "shambase/print.hpp"
#include "shamcmdopt/cmdopt.hpp"
#include "shamcmdopt/details/generic_opts.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/term_colors.hpp"
#include <string_view>
#include <vector>

// modified from https://rosettacode.org/wiki/Check_output_device_is_a_terminal
#if __has_include(<unistd.h>)
    #include <unistd.h>
    #define ISATTY isatty
    #define FILENO fileno
/**
 * @brief Test if current terminal is a tty
 *
 * @return true is a tty
 * @return false is not a tty
 */
bool is_a_tty() { return ISATTY(FILENO(stdout)); }
#else
/**
 * @brief Test if current terminal is a tty
 *
 * @return true is a tty
 * @return false is not a tty
 */
bool is_a_tty() { return true; }
#endif

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
    }

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

    void process_cmdopt_generic_opts() {

        process_colors();

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
        }
    }

} // namespace shamcmdopt
