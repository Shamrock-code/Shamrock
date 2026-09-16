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
 * @file term_colors.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "sham/term/color.hpp"
#include <cstdint>
#include <string>
namespace shambase {

    namespace term_colors {

        /// Enable colors in logs
        inline void enable_colors() { sham::term::enable_colors(); }

        /// Disable all colors
        inline void disable_colors() { sham::term::disable_colors(); }

        /// Are colors enabled
        inline bool colors_enabled() { return sham::term::are_colors_enabled(); }

        /// Get the detected/forced terminal color support level
        inline sham::term::ColorLevel color_level() { return sham::term::color_level(); }

        /// Get the empty terminal escape
        inline const std::string empty() { return ""; };
        /// Get the reset terminal escape char
        inline const std::string reset() { return sham::term::style::reset(); };
        /// Get the bold terminal escape char
        inline const std::string bold() { return sham::term::style::bold(); };
        /// Get the faint terminal escape char
        inline const std::string faint() { return sham::term::style::faint(); };
        /// Get the underline terminal escape char
        inline const std::string underline() { return sham::term::style::underline(); };
        /// Get the blink terminal escape char
        inline const std::string blink() { return sham::term::style::blink(); };
        /// Get the black terminal escape char
        inline const std::string col8b_black() { return sham::term::colors_8b::black(); };
        /// Get the red terminal escape char
        inline const std::string col8b_red() { return sham::term::colors_8b::red(); };
        /// Get the green terminal escape char
        inline const std::string col8b_green() { return sham::term::colors_8b::green(); };
        /// Get the yellow terminal escape char
        inline const std::string col8b_yellow() { return sham::term::colors_8b::yellow(); };
        /// Get the blue terminal escape char
        inline const std::string col8b_blue() { return sham::term::colors_8b::blue(); };
        /// Get the magenta (pink) terminal escape char
        inline const std::string col8b_magenta() { return sham::term::colors_8b::magenta(); };
        /// Get the cyan terminal escape char
        inline const std::string col8b_cyan() { return sham::term::colors_8b::cyan(); };
        /// Get the white terminal escape char
        inline const std::string col8b_white() { return sham::term::colors_8b::white(); };

        /// Get the 256-color palette foreground terminal escape char (empty if unsupported)
        inline std::string col256_foreground(std::uint8_t index) {
            return sham::term::colors_256::foreground(index);
        };

        /// Get the 256-color palette background terminal escape char (empty if unsupported)
        inline std::string col256_background(std::uint8_t index) {
            return sham::term::colors_256::background(index);
        };

        /// Get the 24-bit RGB truecolor foreground terminal escape char (empty if unsupported)
        inline std::string rgb_foreground(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return sham::term::colors_24b::foreground(r, g, b);
        };

        /// Get the 24-bit RGB truecolor background terminal escape char (empty if unsupported)
        inline std::string rgb_background(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
            return sham::term::colors_24b::background(r, g, b);
        };

    } // namespace term_colors

} // namespace shambase
