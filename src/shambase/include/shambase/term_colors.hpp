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
#include <string>
namespace shambase {

    namespace term_colors {

        /// Enable colors in logs
        inline void enable_colors() { sham::term::enable_colors(); }

        /// Disable all colors
        inline void disable_colors() { sham::term::disable_colors(); }

        /// Are colors enabled
        inline bool colors_enabled() { return sham::term::are_colors_enabled(); }

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

    } // namespace term_colors

} // namespace shambase
