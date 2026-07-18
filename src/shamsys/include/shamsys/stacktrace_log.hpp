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
 * @file stacktrace_log.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utilities to generate a backtrace for the crash report
 */

#include <string>

namespace shamsys {

    /**
     * @brief Initialize the backtrace utilities
     * @param enable_colors Whether to enable colors in the backtrace
     */
    void init_backtrace_utilities(bool enable_colors);

    /**
     * @brief Generate a backtrace for the crash report
     * @return std::string The backtrace log (can include profiler stacktrace and true stacktrace)
     */
    std::string crash_report_backtrace();

} // namespace shamsys
