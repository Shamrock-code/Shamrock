// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file color.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Terminal color escape sequence constants and enable/disable control implementation
 *
 */

#include <sham/term/color.hpp>

#define TERM_ESCAPTE_CHAR "\x1b["
namespace {
    bool colors_enabled = true;

    const char *_empty_str     = "";
    const char *_esc_char      = TERM_ESCAPTE_CHAR;
    const char *_reset         = TERM_ESCAPTE_CHAR "0m";
    const char *_bold          = TERM_ESCAPTE_CHAR "1m";
    const char *_faint         = TERM_ESCAPTE_CHAR "2m";
    const char *_underline     = TERM_ESCAPTE_CHAR "4m";
    const char *_blink         = TERM_ESCAPTE_CHAR "5m";
    const char *_col8b_black   = TERM_ESCAPTE_CHAR "30m";
    const char *_col8b_red     = TERM_ESCAPTE_CHAR "31m";
    const char *_col8b_green   = TERM_ESCAPTE_CHAR "32m";
    const char *_col8b_yellow  = TERM_ESCAPTE_CHAR "33m";
    const char *_col8b_blue    = TERM_ESCAPTE_CHAR "34m";
    const char *_col8b_magenta = TERM_ESCAPTE_CHAR "35m";
    const char *_col8b_cyan    = TERM_ESCAPTE_CHAR "36m";
    const char *_col8b_white   = TERM_ESCAPTE_CHAR "37m";
} // namespace

namespace sham::term {

    namespace style {
        const char *reset() { return (colors_enabled) ? _reset : _empty_str; }
        const char *bold() { return (colors_enabled) ? _bold : _empty_str; }
        const char *faint() { return (colors_enabled) ? _faint : _empty_str; }
        const char *underline() { return (colors_enabled) ? _underline : _empty_str; }
        const char *blink() { return (colors_enabled) ? _blink : _empty_str; }
    } // namespace style

    namespace colors_8b {
        const char *black() { return (colors_enabled) ? _col8b_black : _empty_str; }
        const char *red() { return (colors_enabled) ? _col8b_red : _empty_str; }
        const char *green() { return (colors_enabled) ? _col8b_green : _empty_str; }
        const char *yellow() { return (colors_enabled) ? _col8b_yellow : _empty_str; }
        const char *blue() { return (colors_enabled) ? _col8b_blue : _empty_str; }
        const char *magenta() { return (colors_enabled) ? _col8b_magenta : _empty_str; }
        const char *cyan() { return (colors_enabled) ? _col8b_cyan : _empty_str; }
        const char *white() { return (colors_enabled) ? _col8b_white : _empty_str; }
    } // namespace colors_8b

    /// Enable colors
    void enable_colors() { colors_enabled = true; }

    /// Disable all colors
    void disable_colors() { colors_enabled = false; }

    /// Are colors enabled
    bool are_colors_enabled() { return colors_enabled; }

} // namespace sham::term
