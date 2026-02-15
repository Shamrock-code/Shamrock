// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/matrix.hpp"
#include "shammath/matrix_op.hpp"
#include "shammath/solve.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shammath/matrix::mat_inv_33", test_inv_33, 1) {

    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> expected_inverse{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat<f32, 3, 3> result;
    shammath::mat_inv_33(mat.get_mdspan(), result.get_mdspan());

    REQUIRE_EQUAL(result.data, expected_inverse.data);
}

TestStart(Unittest, "shammath/matrix::mat_prod", test_mat_prod, 1) {

    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> inverse{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat<f32, 3, 3> id = shammath::mat_identity<f32, 3>();

    shammath::mat<f32, 3, 3> result;
    shammath::mat_prod(mat.get_mdspan(), inverse.get_mdspan(), result.get_mdspan());

    REQUIRE_EQUAL(result.data, id.data);
}

TestStart(Unittest, "shammath/matrix::mat_prod_vec", test_mat_prod_vec, 1) {

    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::vec<f32, 3> vec{
        // clang-format off
          0,
         -3,
         -2
        // clang-format on
    };

    shammath::vec<f32, 3> result;
    shammath::mat_prod(mat.get_mdspan(), vec.get_mdspan_mat_col(), result.get_mdspan_mat_col());

    shammath::vec<f32, 3> expected_result{
        // clang-format off
         19,
         22,
        -33
        // clang-format on
    };

    REQUIRE_EQUAL(result.data, expected_result.data);
}

TestStart(Unittest, "shammath/matrix::mat_plus", test_mat_plus, 1) {

    shammath::mat<f32, 3, 3> mat1{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> mat2{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat<f32, 3, 3> result;
    shammath::mat_plus(mat1.get_mdspan(), mat2.get_mdspan(), result.get_mdspan());

    shammath::mat<f32, 3, 3> expected_result{
        // clang-format off
          4,  -8, -4,
          6, -10, -4,
        -11,  13,  4
        // clang-format on
    };

    REQUIRE_EQUAL(result.data, expected_result.data);
}

TestStart(Unittest, "shammath/matrix::mat_plus_equal", test_mat_plus_equal, 1) {

    shammath::mat<f32, 3, 3> mat1{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> mat2{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat_plus_equal(mat1.get_mdspan(), mat2.get_mdspan());

    shammath::mat<f32, 3, 3> expected_result{
        // clang-format off
          4,  -8, -4,
          6, -10, -4,
        -11,  13,  4
        // clang-format on
    };

    REQUIRE(mat1 == expected_result);
}

TestStart(Unittest, "shammath/matrix::mat_sub", test_mat_sub, 1) {

    shammath::mat<f32, 3, 3> mat1{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> mat2{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat<f32, 3, 3> result;
    shammath::mat_sub(mat1.get_mdspan(), mat2.get_mdspan(), result.get_mdspan());

    shammath::mat<f32, 3, 3> expected_result{
        // clang-format off
        -4,  2,  0,
        -4,  2,  0,
         5, -5, -2
        // clang-format on
    };

    REQUIRE_EQUAL(result.data, expected_result.data);
}

TestStart(Unittest, "shammath/matrix::mat_sub_equal", test_mat_sub_equal, 1) {

    shammath::mat<f32, 3, 3> mat1{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> mat2{
        // clang-format off
         4, -5, -2,
         5, -6, -2,
        -8,  9,  3
        // clang-format on
    };

    shammath::mat_sub_equal(mat1.get_mdspan(), mat2.get_mdspan());

    shammath::mat<f32, 3, 3> expected_result{
        // clang-format off
        -4,  2,  0,
        -4,  2,  0,
         5, -5, -2
        // clang-format on
    };

    REQUIRE_EQUAL(mat1.data, expected_result.data);
}

TestStart(Unittest, "shammath/matrix::mat_set_identity", test_mat_set_identity, 1) {

    shammath::mat<f32, 3, 3> mat{
        // clang-format off
         0, -3, -2,
         1, -4, -2,
        -3,  4,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> result;
    shammath::mat_set_identity(result.get_mdspan());

    shammath::mat<f32, 3, 3> expected_result{
        // clang-format off
         1,  0,  0,
         0,  1,  0,
         0,  0,  1
        // clang-format on
    };

    REQUIRE_EQUAL(result.data, expected_result.data);
}

TestStart(Unittest, "shammath/matrix::mat_L1_norm", test_mat_L1_norm, 1) {
    shammath::mat<f32, 3, 3> M{
        // clang-format off
         1, -2,  3,
         4,  5,  6,
         7,  8,  9
        // clang-format on
    };

    f32 ex_res = 24, res;
    shammath::mat_L1_norm<f32, f32>(M.get_mdspan(), res);
    REQUIRE_EQUAL(res, ex_res);
}

TestStart(Unittest, "shammath/matrix::mat_set_nul", test_mat_set_nul, 1) {
    shammath::mat<f32, 3, 3> mat;
    shammath::mat<f32, 3, 3> ex_res{
        // clang-format off
         0,  0,  0,
         0,  0,  0,
         0,  0,  0
        // clang-format on
    };
    shammath::mat_set_nul(mat.get_mdspan());
    REQUIRE_EQUAL(mat.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::vec_set_nul", test_vec_set_nul, 1) {
    shammath::vec<f32, 3> v;
    shammath::vec<f32, 3> ex_res{
        // clang-format off
         0,  0,  0,
        // clang-format on
    };
    shammath::vec_set_nul(v.get_mdspan());
    REQUIRE_EQUAL(v.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::vec_copy", test_vec_copy, 1) {
    shammath::vec<f32, 3> v{
        // clang-format off
         1,  2,  3,
        // clang-format on
    };

    shammath::vec<f32, 3> ex_res{
        // clang-format off
         1,  2,  3,
        // clang-format on
    };

    shammath::vec<f32, 3> v_res;

    shammath::vec_copy(v.get_mdspan(), v_res.get_mdspan());
    REQUIRE_EQUAL(v_res.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::vec_axpy_beta", test_vec_axpy_beta, 1) {
    shammath::vec<f32, 3> v1{
        // clang-format off
         1,  0.25,  8,
        // clang-format on
    };

    shammath::vec<f32, 3> v2{
        // clang-format off
         2,  -0.25,  2,
        // clang-format on
    };
    i32 a = -1, b = 1;
    shammath::vec<f32, 3> ex_res{
        // clang-format off
         1,  -0.5,  -6,
        // clang-format on
    };
    shammath::vec_axpy_beta(a, v1.get_mdspan(), b, v2.get_mdspan());
    REQUIRE_EQUAL(v2.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::vec_axpy", test_vec_axpy, 1) {
    shammath::vec<f32, 3> v1{
        // clang-format off
         1,  0.25,  8,
        // clang-format on
    };

    shammath::vec<f32, 3> v2{
        // clang-format off
         2,  -0.25,  2,
        // clang-format on
    };
    i32 a = -1;
    shammath::vec<f32, 3> ex_res{
        // clang-format off
         1,  -0.5,  -6,
        // clang-format on
    };
    shammath::vec_axpy(a, v1.get_mdspan(), v2.get_mdspan());
    REQUIRE_EQUAL(v2.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_axpy_beta", test_mat_axpy_beta, 1) {
    shammath::mat<f64, 3, 3> M{
        // clang-format off
          1,  7,  5,
          5,  3,  4,
         -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> N{
        // clang-format off
          1,  7,    5,
          1,  0.5,  4 ,
         -1,  3.1,  0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         -1.5,  -10.5,  -7.5,
          0.5,    0.5,  -6,
          1.5,   -4.7,  -0.375
        // clang-format on
    };
    const f32 b = 0.5, a = -2;
    shammath::mat_axpy_beta(a, N.get_mdspan(), b, M.get_mdspan());
    REQUIRE_EQUAL(M.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_axpy", test_mat_axpy, 1) {
    shammath::mat<f64, 3, 3> M{
        // clang-format off
          1,  7,  5,
          5,  3,  4,
         -1,  3,  0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> N{
        // clang-format off
         1,  7,   5,
         1,  0.5, 4,
        -1,  3.1, 0.25
        // clang-format on
    };

    shammath::mat<f64, 3, 3> ex_res{
        // clang-format off
         -1, -7,   -5,
          3,  2,   -4,
          1, -3.2, -0.25
        // clang-format on
    };
    const f32 a = -2;
    shammath::mat_axpy(a, N.get_mdspan(), M.get_mdspan());
    REQUIRE_EQUAL(M.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_gemm", test_mat_gemm, 1) {
    shammath::mat<f32, 3, 3> A{
        // clang-format off
         1,  2,  3,
         4,  1, -1,
         0, -1,  0
        // clang-format on
    };

    shammath::mat<f32, 3, 3> B{
        // clang-format off
         0,  3,  0,
         2,  1,  1,
         0, -1,  0
        // clang-format on
    };

    shammath::mat<f32, 3, 3> C{
        // clang-format off
         1,    0,  0.5,
         0.25, 1,  0,
         0,    0,  1
        // clang-format on
    };

    shammath::mat<f32, 3, 3> ex_res{
        // clang-format off
          3,     2,   1.5,
          1.75, 13,   1,
         -2,    -1,  -2
        // clang-format on
    };
    const i32 a = 1, b = -1;
    shammath::mat_gemm(a, A.get_mdspan(), B.get_mdspan(), b, C.get_mdspan());
    REQUIRE_EQUAL(C.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_plus_equal_scalar_id", test_mat_plus_equal_scalar_id, 1) {
    shammath::mat<f32, 3, 3> A{
        // clang-format off
         0,  3,  0,
         2,  1,  1,
         0,  -1,  0
        // clang-format on
    };

    const i32 b = 2;

    shammath::mat<f32, 3, 3> ex_res{
        // clang-format off
         2,  3,  0,
         2,  3,  1,
         0, -1,  2
        // clang-format on
    };
    shammath::mat_plus_equal_scalar_id(A.get_mdspan(), b);
    REQUIRE_EQUAL(A.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_gemv", test_mat_gemv, 1) {
    shammath::mat<f32, 3, 3> B{
        // clang-format off
         1, 2, 3,
         4, 5, 6,
         7, 8, 9
        // clang-format on
    };
    shammath::vec<f32, 3> x{
        // clang-format off
         1, -1, 1
        // clang-format on
    };
    shammath::vec<f32, 3> y{
        // clang-format off
         2, 3, 1
        // clang-format on
    };
    f32 a = 2, b = -0.5;
    shammath::vec<f32, 3> ex_res{
        // clang-format off
         3, 8.5, 15.5
        // clang-format on
    };
    shammath::mat_gemv(a, B.get_mdspan(), x.get_mdspan(), b, y.get_mdspan());
    REQUIRE_EQUAL(y.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::mat_transpose", test_transpose, 1) {
    shammath::mat<f32, 2, 3> A{
        // clang-format off
        1,2,3,
        4,5,6,
        // clang-format on
    };
    shammath::mat<f32, 3, 2> B;
    shammath::mat<f32, 3, 2> ex_res{
        // clang-format off
        1, 4,
        2, 5,
        3, 6,
        // clang-format on
    };
    shammath::mat_transpose(A.get_mdspan(), B.get_mdspan());
    REQUIRE_EQUAL(B.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::Cholesky_decomp", test_Cholesky_decomp, 1) {
    shammath::mat<f32, 4, 4> M{
        // clang-format off
         1, 1, 1, 1,
         1, 5, 5, 5,
         1, 5, 14, 14,
        1, 5, 14, 15,
        // clang-format on
    };
    shammath::mat<f32, 4, 4> L;
    shammath::mat<f32, 4, 4> ex_res{
        // clang-format off
        1,0,0,0,
        1,2,0,0,
        1,2,3,0,
        1,2,3,1
        // clang-format on
    };
    shammath::Cholesky_decomp(M.get_mdspan(), L.get_mdspan());
    REQUIRE_EQUAL(L.data, ex_res.data);
}

TestStart(Unittest, "shammath/matrix::Cholesky_solve", test_Cholesky_solve, 1) {
    shammath::mat<f32, 3, 3> M{
        // clang-format off
        6,15,55,
        15,55,225,
        55,225,979
        // clang-format on
    };

    shammath::vec<f32, 3> y{
        // clang-format off
        76,295,1259
        // clang-format on
    };

    shammath::vec<f32, 3> x;
    shammath::vec<f32, 3> ex_res{
        // clang-format off
        1,1,1
        // clang-format on
    };
    shammath::Cholesky_solve(M.get_mdspan(), y.get_mdspan(), x.get_mdspan());
    REQUIRE_EQUAL_CUSTOM_COMP_NAMED("", x.data, ex_res.data, [](const auto &p1, const auto &p2) {
        return sycl::pow(p1[0] - p2[0], 2) + sycl::pow(p1[1] - p2[1], 2)
                   + sycl::pow(p1[2] - p2[2], 2)
               < 1e-9;
    });
}

// This test uses Eckerle4 from NIST Standard Reference Database [Eckerle, K., NIST (1979)]
// https://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml
TestStart(Unittest, "shammath/solve::least_squares", test_least_squares, 1) {
    std::vector<f64> p0 = {1, 1e1, 5e2};
    std::vector<f64> X
        = {400.000000e0, 405.000000e0, 410.000000e0, 415.000000e0, 420.000000e0, 425.000000e0,
           430.000000e0, 435.000000e0, 436.500000e0, 438.000000e0, 439.500000e0, 441.000000e0,
           442.500000e0, 444.000000e0, 445.500000e0, 447.000000e0, 448.500000e0, 450.000000e0,
           451.500000e0, 453.000000e0, 454.500000e0, 456.000000e0, 457.500000e0, 459.000000e0,
           460.500000e0, 462.000000e0, 463.500000e0, 465.000000e0, 470.000000e0, 475.000000e0,
           480.000000e0, 485.000000e0, 490.000000e0, 495.000000e0, 500.000000e0};

    std::vector<f64> Y = {
        0.0001575e0, 0.0001699e0, 0.0002350e0, 0.0003102e0, 0.0004917e0, 0.0008710e0, 0.0017418e0,
        0.0046400e0, 0.0065895e0, 0.0097302e0, 0.0149002e0, 0.0237310e0, 0.0401683e0, 0.0712559e0,
        0.1264458e0, 0.2073413e0, 0.2902366e0, 0.3445623e0, 0.3698049e0, 0.3668534e0, 0.3106727e0,
        0.2078154e0, 0.1164354e0, 0.0616764e0, 0.0337200e0, 0.0194023e0, 0.0117831e0, 0.0074357e0,
        0.0022732e0, 0.0008800e0, 0.0004579e0, 0.0002345e0, 0.0001586e0, 0.0001143e0, 0.0000710e0};
    auto ls = shammath::least_squares(
        [](std::vector<f64> p, f64 x) -> f64 {
            f64 b1 = p[0];
            f64 b2 = p[1];
            f64 b3 = p[2];
            return (b1 / b2) * exp(-0.5 * ((x - b3) / b2) * ((x - b3) / b2));
        },
        X,
        Y,
        p0,
        1000,
        1e-9);

    auto pfit                     = ls.first;
    auto R2                       = ls.second;
    std::vector<f64> res          = {pfit[0], pfit[1], pfit[2], R2};
    std::vector<f64> ex_res       = {1.55, 4.08, 4.5154e2};
    std::vector<f64> ex_deviation = {2e-2, 4.7e-2, 4.7e-2, 1e-2};

    bool test_fit = true;
    for (size_t i; i < 4; i++) {
        if (sham::abs(res[i] - ex_res[i]) > ex_deviation[i]) {
            test_fit = false;
        }
    };
    REQUIRE(test_fit);
}
