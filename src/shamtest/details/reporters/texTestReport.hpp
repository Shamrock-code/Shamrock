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
 * @file texTestReport.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtest/details/TestResult.hpp"
#include <vector>

namespace shamtest::details {

    /**
     * @brief Make a tex report from the list of test results
     *
     * @param results
     * @param mark_fail
     * @return std::string
     */
    std::string make_test_report_tex(std::vector<TestResult> &results, bool mark_fail);

} // namespace shamtest::details
