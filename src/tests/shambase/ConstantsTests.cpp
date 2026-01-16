// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/constants.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/constants", checkconstantmatchsycl, 1) {

    using namespace shambase::constants;

    REQUIRE_FLOAT_EQUAL(pi<f32>, 4 * sycl::atan(unity<f32>), 1e-25);
    REQUIRE_FLOAT_EQUAL(pi<f64>, 4 * sycl::atan(unity<f64>), 1e-25);
}
