// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file tty.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains implementations of tty info getters
 *
 */

#include "shamcmdopt/tty.hpp"
#include <cstdio>
#include <utility>

#if __has_include(<unistd.h>)
    #include <unistd.h>
#endif

#if __has_include(<sys/ioctl.h>)
    #include <sys/ioctl.h>
#endif

namespace shamcmdopt {

    bool is_a_tty() {
#if __has_include(<unistd.h>)
        return isatty(fileno(stdout));
#else
        return true;
#endif
    }

    /// Forced width of the terminal, if set by set_tty_columns.
    /// If set to 0, get_tty_dim returns the actual width of the terminal.
    u32 tty_forced_width = 0;

    void set_tty_columns(u32 columns) { tty_forced_width = columns; }

    /**
     * @brief Get the number of columns and lines of the current terminal
     *
     * If forced width is set (by set_tty_columns), this function returns the forced width.
     * If the current terminal is not a tty, the function returns {10, 100}.
     *
     * @return Number of columns and lines if the current terminal
     */
    std::pair<u32, u32> get_tty_dim() {
        // If forced width is set, return it
        if (tty_forced_width > 0) {
            return {10, tty_forced_width};
        }

        // If the current terminal is not a tty, return a default value
        if (!is_a_tty()) {
            return {10, 100};
        }

        // If the terminal is a tty, query its size
#if __has_include(<sys/ioctl.h>) &&  __has_include(<unistd.h>)
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        // If the terminal size is invalid, return a default value
        if (w.ws_col == 0 || w.ws_row == 0) {
            return {10, 100};
        }
        return {w.ws_row, w.ws_col};
#else
        // If the terminal is a tty but we don't have a way to query its size,
        // return a default value
        return {10, 100};
#endif
    }

    u32 get_tty_columns() { return get_tty_dim().second; }
    u32 get_tty_lines() { return get_tty_dim().first; }

} // namespace shamcmdopt
