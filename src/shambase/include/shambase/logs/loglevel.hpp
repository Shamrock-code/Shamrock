// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file loglevel.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"

/**
 * @brief Namespace containing logs utils
 */
namespace shambase::logs {
    /**
     * @namespace details
     * @brief Namespace for internal details of the logs module
     */
    namespace details {
        /**
         * @brief Internal variable to store the global log level
         */
        inline i8 loglevel = 0;
    } // namespace details

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Log level manip
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Set the global log level
     *
     * @param val The new log level
     */
    inline void set_loglevel(i8 val) { details::loglevel = val; }

    /**
     * @brief Get the current global log level
     *
     * @return The current log level
     */
    inline i8 get_loglevel() { return details::loglevel; }
} // namespace shambase::logs
