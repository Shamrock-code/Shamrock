// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

} // namespace shamcmdopt
