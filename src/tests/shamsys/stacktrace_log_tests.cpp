// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file stacktrace_log_tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamcomm/logs.hpp"
#include "shamsys/stacktrace_log.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamsys/stacktrace_log", stacktrace_log_tests, -1) {
    shamcomm::logs::raw_ln(shamsys::crash_report_backtrace());
}
