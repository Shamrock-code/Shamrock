// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file texTestReport.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
