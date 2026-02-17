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
 * @file generic_opts.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file handler generic cli & env options
 *
 */

namespace shamcmdopt {

    /**
     * @brief Register generic cli and env variables options
     *
     */
    void register_cmdopt_generic_opts();

    /**
     * @brief Process generic cli and env variables options
     */
    void process_cmdopt_generic_opts();

    /**
     * @brief Print the help message.
     */
    void print_help();

    /**
     * @brief Check if the current terminal supports UTF-8 encoding
     *
     * This function checks if the current terminal is a TTY and if it supports UTF-8.
     * It examines the locale settings (LC_ALL, LC_CTYPE, LANG) to determine UTF-8 support.
     *
     * @return true if UTF-8 is supported, false otherwise
     */
    bool check_utf8_support();

} // namespace shamcmdopt
