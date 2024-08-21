// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file tty.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

    u32 tty_forced_width = 0;
    void set_tty_columns(u32 columns) { tty_forced_width = columns; }

    std::pair<u32, u32> get_tty_dim() {
        if (tty_forced_width > 0) {
            return {10, tty_forced_width};
        }
        if (!is_a_tty()) {
            return {10, 100};
        }
#if __has_include(<sys/ioctl.h>) &&  __has_include(<unistd.h>)
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

        if(w.ws_col == 0 || w.ws_row == 0) {
            return {10, 100};
        }
        return {w.ws_row, w.ws_col};
#else
        return {10, 100};
#endif
    }

    u32 get_tty_columns() { return get_tty_dim().second; }
    u32 get_tty_lines() { return get_tty_dim().first; }

} // namespace shamcmdopt
