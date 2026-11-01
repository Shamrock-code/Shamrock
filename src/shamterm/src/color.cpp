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
#include <string>

#define TERM_ESCAPTE_CHAR "\x1b["
namespace {
    /// Currently set terminal color support level, as set by sham::term::set_color_level.
    /// This variable should only be accessible through color_level() and set_color_level()
    sham::term::ColorLevel color_level_value = sham::term::ColorLevel::NoColor;

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

    ColorLevel color_level() { return color_level_value; }
    void set_color_level(ColorLevel level) { color_level_value = level; }

    /// Enable colors
    /// TODO: should set to the result of the color max(color_detect(), Basic)
    void enable_colors() { set_color_level(ColorLevel::Basic); }

    /// Disable all colors
    void disable_colors() { set_color_level(ColorLevel::NoColor); }

    /// Are colors enabled
    bool are_colors_enabled() { return color_level() != ColorLevel::NoColor; }

    namespace style {
        const char *reset() { return are_colors_enabled() ? _reset : _empty_str; }
        const char *bold() { return are_colors_enabled() ? _bold : _empty_str; }
        const char *faint() { return are_colors_enabled() ? _faint : _empty_str; }
        const char *underline() { return are_colors_enabled() ? _underline : _empty_str; }
        const char *blink() { return are_colors_enabled() ? _blink : _empty_str; }
    } // namespace style

    namespace colors_8b {
        const char *black() { return are_colors_enabled() ? _col8b_black : _empty_str; }
        const char *red() { return are_colors_enabled() ? _col8b_red : _empty_str; }
        const char *green() { return are_colors_enabled() ? _col8b_green : _empty_str; }
        const char *yellow() { return are_colors_enabled() ? _col8b_yellow : _empty_str; }
        const char *blue() { return are_colors_enabled() ? _col8b_blue : _empty_str; }
        const char *magenta() { return are_colors_enabled() ? _col8b_magenta : _empty_str; }
        const char *cyan() { return are_colors_enabled() ? _col8b_cyan : _empty_str; }
        const char *white() { return are_colors_enabled() ? _col8b_white : _empty_str; }
    } // namespace colors_8b

    namespace colors_256 {
        namespace {
            /// Build a \x1b[<mode>;5;Nm 256-color escape sequence, or an empty string if
            /// color_level_value is set below ColorLevel::ANSI256.
            std::string build(int mode, std::uint8_t index) {
                if (color_level_value < ColorLevel::ANSI256) {
                    return "";
                }
                return std::string(TERM_ESCAPTE_CHAR) + std::to_string(mode) + ";5;"
                       + std::to_string(index) + "m";
            }
        } // namespace

        std::string foreground(std::uint8_t index) { return build(38, index); }
        std::string background(std::uint8_t index) { return build(48, index); }
    } // namespace colors_256

    namespace colors_24b {
        namespace {
            /// Build a \x1b[<mode>;2;r;g;bm truecolor escape sequence, or an empty string if
            /// color_level_value is set below ColorLevel::TrueColor.
            std::string build(int mode, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
                if (color_level_value < ColorLevel::TrueColor) {
                    return "";
                }
                return std::string(TERM_ESCAPTE_CHAR) + std::to_string(mode) + ";2;"
                       + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b)
                       + "m";
            }
        } // namespace

        std::string foreground(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return build(38, r, g, b);
        }
        std::string background(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return build(48, r, g, b);
        }
    } // namespace colors_24b

} // namespace sham::term
