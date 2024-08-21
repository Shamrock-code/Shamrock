// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file tty.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file contains tty info getters
 *
 */

#include "shambase/aliases_int.hpp"

namespace shamcmdopt {

    /**
     * @brief Test if current terminal is a tty
     *
     * @return true is a tty
     * @return false is not a tty
     */
    bool is_a_tty();

    /**
     * @brief Get the number of columns of the current terminal
     *
     * If forced width is set (by set_tty_columns), this function returns the forced width.
     * If the current terminal is not a tty, the function returns 100.
     *
     * @return Number of columns if the current terminal
     */
    u32 get_tty_columns();

    /**
     * @brief Get the number of lines of the current terminal
     *
     * If the current terminal is not a tty, the function returns 10.
     *
     * @return Number of lines if the current terminal
     */
    u32 get_tty_lines();

    /**
     * @brief Set the forced width of the terminal
     *
     * By default, the function get_tty_columns() returns the number of columns of the current
     * terminal. If this function is called with a non-zero value, get_tty_columns() will always
     * return this value until set_tty_columns() is called again with a different value.
     *
     * @param columns The width of the terminal. If zero, the default behavior of get_tty_columns()
     * is restored.
     */
    void set_tty_columns(u32 columns);

    /**
     * @brief Reset the forced width of the terminal
     *
     * Calls set_tty_columns(0) to restore the default behavior of get_tty_columns().
     */
    inline void reset_tty_columns() { set_tty_columns(0); }
} // namespace shamcmdopt
