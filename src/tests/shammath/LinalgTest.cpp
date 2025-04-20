// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file LinalgTest.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shammath/LinalgUtilities_mdspan.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/Linalg::mat_set_identity", test_mat_set_identity, 1) {
    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         0, 0, 0,
         0, 0, 0,
        0,  0,  0
        // clang-format on
    };

    shammath::mat<f32, 3, 3> identity_mat{
        // clang-format off
         1, 0, -0,
         0, 1, 0,
        0,  0,  1
        // clang-format on
    };
    shammath::mat_set_identity(mat.get_mdspan());

    REQUIRE(mat == identity_mat);
}

TestStart(Unittest, "shammath/Linalg::mat_copy", test_mat_copy, 1) {
    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         1, 7, 5,
         5, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f32, 3, 3> ex_res{
        // clang-format off
         1, 7, 5,
         5, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f32, 3, 3> res;
    shammath::mat_copy(mat.get_mdspan(), res.get_mdspan());
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::vec_copy", test_vec_copy, 1) {
    shammath::vect<f32, 5> v{
        // clang-format off
        1, 2, 5,
        -5, 3
        // clang-format on
    };

    shammath::vect<f32, 5> ex_res{
        // clang-format off
        1, 2, 5,
        -5, 3
        // clang-format on
    };

    shammath::vect<f32, 5> res;
    shammath::vec_copy(v.get_mdspan(), res.get_mdspan());
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_set_nul", test_mat_set_nul, 1) {
    shammath::mat<f32, 3, 3> mat;
    shammath::mat<f32, 3, 3> ex_res{
        // clang-format off
         0, 0, 0,
         0, 0, 0,
        0,  0,  0
        // clang-format on
    };
    shammath::mat_set_nul(mat.get_mdspan());
    REQUIRE(mat == ex_res);
}

TestStart(Unittest, "shammath/Linalg::set_vec_nul", test_set_nul_vec, 1) {
    shammath::vect<f32, 3> v;
    shammath::vect<f32, 3> ex_res{
        // clang-format off
         0, 0, 0
        // clang-format on
    };
    shammath::vec_set_nul(v.get_mdspan());
    REQUIRE(v == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_daxpy", test_mat_daxpy, 1) {
    shammath::mat<f64, 3, 3> M1{
        // clang-format off
         1, 7, 5,
         5, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> M2{
        // clang-format off
         1, 7, 5,
         1, 0.5, 4,
        -1,  3.1,  0.25
        // clang-format on
    };
    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         -1.5, -10.5, -7.5,
         0.5, 0.5, -6,
        1.5,  -4.7,  -0.375
        // clang-format on
    };

    const f32 a = 0.5, b = -2;
    shammath::mat_daxpy(M1.get_mdspan(), M2.get_mdspan(), a, b);
    REQUIRE(M2 == ex_res);
}

TestStart(Unittest, "shammath/Linalg::vec_daxpy", test_vec_daxpy, 1) {
    shammath::vect<f64, 5> v1{
        // clang-format off
         1, 0.25, 8
        // clang-format on
    };

    shammath::vect<f64, 5> v2{
        // clang-format off
         0.33, -0.25, 0.4
        // clang-format on
    };

    f32 a = -0.5, b = 1.;
    shammath::vect<f64, 5> ex_res{
        // clang-format off
         -0.67, -0.375, -3.6
        // clang-format on
    };
    shammath::vec_daxpy(v1.get_mdspan(), v2.get_mdspan(), a, b);
    REQUIRE(v2 == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_gemm", test_mat_gemm, 1) {
    shammath::mat<f64, 3, 3> M1{
        // clang-format off
         1, 2, 5,
         1, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> M2{
        // clang-format off
         1, 3, -2,
         2, 1, 4,
        -1,3,0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         0,-40,-14.5,
         -6,-36,-22,
         -9.5,-1.5,-28.125
        // clang-format on
    };
    shammath::mat<f64, 3, 3> res;
    f32 a = -1, b = 2;
    shammath::mat_gemm(M1.get_mdspan(), M2.get_mdspan(), res.get_mdspan(), a, b);
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::vec_scal", test_vec_scal, 1) {
    shammath::vect<f64, 3> v{1, -2, 4};
    i32 a = -2;
    shammath::vect<f64, 3> ex_res{-2, 4, -8};
    shammath::vect<f64, 3> res;
    shammath::vec_scal(v.get_mdspan(), res.get_mdspan(), a);
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_scal", test_mat_scal, 1) {
    shammath::mat<f64, 3, 3> M1{
        // clang-format off
         1, 2, 5,
         1, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };

    i32 a = -1;
    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         -1, -2, -5,
         -1, -3, -4,
        1,  -3,  -0.25
        // clang-format on
    };
    shammath::mat<f64, 3, 3> res;
    shammath::mat_scal(M1.get_mdspan(), res.get_mdspan(), a);
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_add_scal_id", test_mat_scal_id, 1) {
    shammath::mat<f64, 3, 3> M1{
        // clang-format off
         1, 2, 5,
         1, 3, 4,
        -1,  3,  0.25
        // clang-format on
    };
    i32 b = 2;
    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         3, 2, 5,
         1, 5, 4,
        -1,  3,  2.25
        // clang-format on
    };
    shammath::mat<f64, 3, 3> res;
    shammath::mat_add_scal_id(M1.get_mdspan(), res.get_mdspan(), b);
    REQUIRE(res == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_gemv", test_mat_gemv, 1) {
    shammath::mat<f64, 3, 3> M{
        // clang-format off
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
        // clang-format on
    };
    f32 a = 2, b = -0.5;
    shammath::vect<f64, 3> x{1, -1, 1};
    shammath::vect<f64, 3> y{2, 3, 1};
    shammath::vect<f64, 3> ex_res{3, 8.5, 15.5};

    shammath::mat_gemv(M.get_mdspan(), x.get_mdspan(), y.get_mdspan(), a, b);
    REQUIRE(y == ex_res);
}

TestStart(Unittest, "shammath/Linalg::mat_L1_norm", test_mat_L1_norm, 1) {
    shammath::mat<f64, 3, 3> M{
        // clang-format off
        1, -2, 3,
        4, 5, 6,
        7, 8, 9
        // clang-format on
    };
    f64 ex_res = 24;
    f64 res;
    shammath::mat_L1_norm(M.get_mdspan(), res);
    REQUIRE_EQUAL(ex_res, res);
}
