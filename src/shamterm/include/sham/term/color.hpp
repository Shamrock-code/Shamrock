// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file color.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Terminal color escape sequences and enable/disable control
 *
 */

#include <cstdint>
#include <string>

namespace sham::term {

    /**
     * @brief The possible levels of terminal color support.
     */
    enum class ColorLevel {
        /// No color support, plain ASCII output only.
        NoColor = 0,
        /// ANSI/16 colors (basic SGR codes), \x1b[3Xm.
        Basic = 1,
        /// 256-color palette (16 ANSI + 216-color cube + 24 grayscale), \x1b[38;5;Nm.
        ANSI256 = 2,
        /// 24-bit RGB truecolor, \x1b[38;2;r;g;bm.
        TrueColor = 3,
    };

    /**
     * @brief Query the currently detected/forced terminal color support level.
     */
    ColorLevel color_level();

    /**
     * @brief Set the terminal color support level.
     *
     * @param level the new color support level
     */
    void set_color_level(ColorLevel level);

    /// Enable colors
    void enable_colors();

    /// Disable all colors
    void disable_colors();

    /// Query whether terminal color output is currently enabled.
    bool are_colors_enabled();

    /**
     * @brief Terminal text styling escape sequences (bold, faint, underline, blink, reset).
     */
    namespace style {
        /// escape sequence to reset the terminal text formatting.
        const char *reset();
        /// escape sequence to set bold text formatting.
        const char *bold();
        /// escape sequence to set faint (dim) text formatting.
        const char *faint();
        /// escape sequence to set underlined text formatting.
        const char *underline();
        /// escape sequence to set blinking text formatting.
        const char *blink();
    } // namespace style

    /**
     * @brief 8 bit 8-color palette escape sequences.
     */
    namespace colors_8b {
        /// Escape sequence to set black text color.
        const char *black();
        /// Escape sequence to set red text color.
        const char *red();
        /// Escape sequence to set green text color.
        const char *green();
        /// Escape sequence to set yellow text color.
        const char *yellow();
        /// Escape sequence to set blue text color.
        const char *blue();
        /// Escape sequence to set magenta (pink) text color.
        const char *magenta();
        /// Escape sequence to set cyan text color.
        const char *cyan();
        /// Escape sequence to set white text color.
        const char *white();
    } // namespace colors_8b

    /**
     * @brief 256-color palette escape sequences (\x1b[38;5;Nm / \x1b[48;5;Nm).
     *
     * These are only emitted if colors are enabled (see enable_colors/disable_colors) and
     * color_level() is at least ColorLevel::ANSI256; otherwise an empty string is returned.
     */
    namespace colors_256 {
        /// Escape sequence to set the given palette index as foreground text color.
        std::string foreground(std::uint8_t index);
        /// Escape sequence to set the given palette index as background color.
        std::string background(std::uint8_t index);
    } // namespace colors_256

    /**
     * @brief 24-bit RGB truecolor escape sequences (\x1b[38;2;r;g;bm / \x1b[48;2;r;g;bm).
     *
     * These are only emitted if colors are enabled (see enable_colors/disable_colors) and
     * color_level() is at least ColorLevel::TrueColor; otherwise an empty string is returned.
     */
    namespace colors_24b {
        /// Escape sequence to set the given RGB foreground text color.
        std::string foreground(std::uint8_t r, std::uint8_t g, std::uint8_t b);
        /// Escape sequence to set the given RGB background color.
        std::string background(std::uint8_t r, std::uint8_t g, std::uint8_t b);
    } // namespace colors_24b

} // namespace sham::term
