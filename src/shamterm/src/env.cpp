// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file env.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Terminal color and UTF-8 support detection, and COLUMN parsing from environment variables
 *
 */

#include "sham/term/env.hpp"
#include "sham/term/color.hpp"
#include "sham/term/tty.hpp"
#include <string_view>
#include <cctype>
#include <vector>

namespace {

    /**
     * @brief List of known terminal ident that support colors
     */
    static const std::vector<std::string_view> color_support_term{
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
    bool term_support_color(sham::term::TermEnvVars vars) {

        if (vars.TERM) {
            for (auto term : color_support_term) {
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

    /**
     * @brief Case-insensitive substring search
     *
     * @return true if needle is found in haystack, ignoring case
     */
    bool string_contains_ci(std::string_view haystack, std::string_view needle) {
        if (needle.size() > haystack.size()) {
            return false;
        }
        for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
            bool match = true;
            for (size_t j = 0; j < needle.size(); ++j) {
                if (std::tolower(static_cast<unsigned char>(haystack[i + j]))
                    != std::tolower(static_cast<unsigned char>(needle[j]))) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Detect whether a locale value string denotes a UTF-8 charset
     *
     * @return true if the value contains "utf-8" or "utf8" (case-insensitive)
     */
    bool contains_utf8_marker(std::string_view value) {
        return string_contains_ci(value, "utf-8") || string_contains_ci(value, "utf8");
    }

    /**
     * @brief detect if the current locale supports UTF-8 output
     *
     * Follows POSIX LC_CTYPE category precedence: LC_ALL overrides LC_CTYPE overrides LANG.
     * LANGUAGE is a gettext message-language list, not a charset setting, so it is not used here.
     *
     * @return true
     * @return false
     */
    bool term_support_utf8(sham::term::TermEnvVars vars) {
        if (vars.lc_all) {
            return contains_utf8_marker(*vars.lc_all);
        }
        if (vars.lc_ctype) {
            return contains_utf8_marker(*vars.lc_ctype);
        }
        if (vars.LANG) {
            return contains_utf8_marker(*vars.LANG);
        }
        return false;
    }

} // namespace

namespace sham::term {

    void parse_terminal_support(TermEnvVars vars, const term_parse_callback_t &error_callback) {
        if (term_support_color(vars)) {
            enable_colors();
        } else {
            disable_colors();
        }

        bool has_envvar_nocolor = bool(vars.NO_COLOR);
        bool has_envvar_color   = bool(vars.CLICOLOR_FORCE);

        if (has_envvar_color && has_envvar_nocolor) {
            throw error_callback(
                "one can not set both NO_COLOR and CLICOLOR_FORCE",
                std::source_location::current());
        }

        if (has_envvar_nocolor) {
            disable_colors();
        }

        if (has_envvar_color) {
            enable_colors();
        }

        sham::term::set_support_utf8(term_support_utf8(vars));

        bool has_envvar_no_utf8    = bool(vars.NO_UTF8);
        bool has_envvar_force_utf8 = bool(vars.FORCE_UTF8);

        if (has_envvar_no_utf8 && has_envvar_force_utf8) {
            throw error_callback(
                "one can not set both NO_UTF8 and FORCE_UTF8", std::source_location::current());
        }

        if (has_envvar_no_utf8) {
            sham::term::set_support_utf8(false);
        }

        if (has_envvar_force_utf8) {
            sham::term::set_support_utf8(true);
        }

        auto &res = vars.COLUMN;

        int min_sz = 10;
        if (res) {
            try {
                int val = std::stoi(std::string(*res));
                if (val < min_sz) {
                    val = min_sz;
                }
                sham::term::set_tty_columns(val);
            } catch (const std::invalid_argument &a) {
                throw error_callback(
                    "Error : COLUMN is not an integer", std::source_location::current());
            } catch (const std::out_of_range &a) {
                throw error_callback(
                    "Error : COLUMN is out of range", std::source_location::current());
            }
        }
    }

} // namespace sham::term
