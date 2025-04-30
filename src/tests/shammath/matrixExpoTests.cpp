// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/math.hpp"
#include "shammath/matrix.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammath/matrix_op.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <iomanip>

TestStart(Unittest, "shammath/matrix_exp", test_mat_exp, 1) {

    shammath::mat<f64, 3, 3> A{
        // clang-format off
        -0.075, 0.025, 0.05,
        0.025, -0.025, 0,
        0.05,  0,  -0.05
        // clang-format on
    };
    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
        0.92920808533653909134, 0.023795539414865101574, 0.046996375248595824436,
        0.023795539414865101574, 0.975609757004000544, 0.0005947035811343781694,
        0.046996375248595824436, 0.0005947035811343781694, 0.95240892117026976216
        // clang-format on
    };

    shammath::mat<f64, 3, 3> B, F, I, Id;
    i32 K = 9, size_A = 3;
    shammath::mat_exp<f64, f64>(
        K, A.get_mdspan(), F.get_mdspan(), B.get_mdspan(), I.get_mdspan(), Id.get_mdspan(), size_A);
    REQUIRE_EQUAL(A.equal_at_precision(ex_res, 1e-10), true);
}
