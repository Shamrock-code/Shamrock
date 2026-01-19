// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shammath/matrix.hpp"
#include "shammath/matrix_op.hpp"
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
