// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file color_env.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file handler generic cli & env options
 *
 */

#include "sham/term/color_env.hpp"
#include "sham/term/color.hpp"
#include <source_location>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace {

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
    bool term_support_color(sham::term::TermSupportEnvVars vars) {

        if (vars.TERM) {
            for (auto term : color_suport_term) {
                if (*vars.TERM == term) {
                    return true;
                }
            }
        }

        if (vars.COLORTERM) {
            if (*vars.COLORTERM == "truecolor") {
                return true;
            }
            if (*vars.COLORTERM == "24bit") {
                return true;
            }
        }

        return false;
    }

} // namespace

namespace sham::term {

    void parse_terminal_support(TermSupportEnvVars vars) {
        if (term_support_color(vars)) {
            enable_colors();
        } else {
            disable_colors();
        }

        bool has_envvar_nocolor = bool(vars.NO_COLOR);
        bool has_envvar_color   = bool(vars.CLICOLOR_FORCE);

        if (has_envvar_color && has_envvar_nocolor) {
            throw std::runtime_error("one can not set both NO_COLOR and CLICOLOR_FORCE");
        }

        if (has_envvar_nocolor) {
            disable_colors();
        }

        if (has_envvar_color) {
            enable_colors();
        }
    }

} // namespace sham::term
