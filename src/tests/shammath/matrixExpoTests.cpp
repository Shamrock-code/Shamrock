// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/matrix.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/matrix_expo::scaling_and_squaring", test_mat_expo, 1) {

    shammath::mat<f32, 3, 3> A{
        // clang-format off
        -0.075, 0.025, 0.05,
        0.025, -0.025, 0,
        0.05,  0,  -0.05
        // clang-format on
    };
    shammath::mat<f32, 3, 3> B, F, I, Id;
    i32 K = 9, size_A = 3;
    shammath::mat_expo<f32, f32>(
        K, A.get_mdspan(), F.get_mdspan(), B.get_mdspan(), I.get_mdspan(), Id.get_mdspan(), size_A);

    // std::cout << "\n\n " << "=============== exponential of A ================= " << "\n\n";
    // for (int i = 0; i < size_A; i++) {
    //     std::cout << "line ( " << i << " )" << "\n\n";
    //     for (int j = 0; j < size_A; j++) {
    //         std::cout << "colonne ( " << i << ", " << j << " )" << A.get_mdspan()(i, j) <<
    //         "\n\n";
    //     }

    //     std::cout << "\n\n";
    // }
}
