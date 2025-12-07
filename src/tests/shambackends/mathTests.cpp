// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambackends/math.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambackends/math.hpp:roundup_pow2_clz", shambackendsmathroundup_pow2_clz, 1) {

    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(0), 1);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(1), 1);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(2), 2);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(3), 4);

    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(2147483647), 2147483648);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(2147483648), 2147483648);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(2147483649), 0);
    REQUIRE_EQUAL(sham::roundup_pow2_clz<u32>(4294967295), 0);
}

inline f64 nan_val = std::numeric_limits<f64>::quiet_NaN();

TestStart(Unittest, "shambackends/math.hpp:inv_sat", shambackendsmathinv_sat, 1) {

    REQUIRE_EQUAL(sham::inv_sat<f64>(1._f64), 1._f64 / 1._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(2._f64), 1._f64 / 2._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(3._f64), 1._f64 / 3._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(4._f64), 1._f64 / 4._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(5._f64), 1._f64 / 5._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(6._f64), 1._f64 / 6._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(7._f64), 1._f64 / 7._f64);

    REQUIRE_EQUAL(sham::inv_sat<f64>(100._f64), 1._f64 / 100._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(1.e-9_f64), 1._f64 / 1.e-9_f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(1.e-10_f64), 0._f64);

    REQUIRE_EQUAL(sham::inv_sat<f64>(-1._f64), -1._f64 / 1._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-2._f64), -1._f64 / 2._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-3._f64), -1._f64 / 3._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-4._f64), -1._f64 / 4._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-5._f64), -1._f64 / 5._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-6._f64), -1._f64 / 6._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-7._f64), -1._f64 / 7._f64);

    REQUIRE_EQUAL(sham::inv_sat<f64>(-100._f64), -1._f64 / 100._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-1.e-9_f64), -1._f64 / 1.e-9_f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(-1.e-10_f64), 0._f64);

    REQUIRE_EQUAL(sham::inv_sat<f64>(0._f64), 0._f64);
    REQUIRE_EQUAL(sham::inv_sat<f64>(nan_val), 0._f64);
}

TestStart(Unittest, "shambackends/math.hpp:inv_sat_positive", shambackendsmathinv_satpos, 1) {

    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(1._f64), 1._f64 / 1._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(2._f64), 1._f64 / 2._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(3._f64), 1._f64 / 3._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(4._f64), 1._f64 / 4._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(5._f64), 1._f64 / 5._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(6._f64), 1._f64 / 6._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(7._f64), 1._f64 / 7._f64);

    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(100._f64), 1._f64 / 100._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(1.e-9_f64), 1._f64 / 1.e-9_f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(1.e-10_f64), 0._f64);

    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(0._f64), 0._f64);
    REQUIRE_EQUAL(sham::inv_sat_positive<f64>(nan_val), 0._f64);
}

TestStart(Unittest, "shambackends/math.hpp:inv_sat_zero", shambackendsmathinv_satzero, 1) {

    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(1._f64), 1._f64 / 1._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(2._f64), 1._f64 / 2._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(3._f64), 1._f64 / 3._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(4._f64), 1._f64 / 4._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(5._f64), 1._f64 / 5._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(6._f64), 1._f64 / 6._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(7._f64), 1._f64 / 7._f64);

    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(100._f64), 1._f64 / 100._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(1.e-9_f64), 1._f64 / 1.e-9_f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(0), 0._f64);

    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-1._f64), -1._f64 / 1._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-2._f64), -1._f64 / 2._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-3._f64), -1._f64 / 3._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-4._f64), -1._f64 / 4._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-5._f64), -1._f64 / 5._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-6._f64), -1._f64 / 6._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-7._f64), -1._f64 / 7._f64);

    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-100._f64), -1._f64 / 100._f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(-1.e-9_f64), -1._f64 / 1.e-9_f64);
    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(0), 0._f64);

    REQUIRE_EQUAL(sham::inv_sat_zero<f64>(nan_val), 0._f64);
}

template<typename T>
T __attribute__((noinline)) bitshift_noinline(T x, T amount) {
    return x << amount;
}

TestStart(Unittest, "shambackends/math.hpp:log2_pow2_num", shambackendsmathlog2_pow2_num, 1) {

    // we have to prevent inline otherwise optimisation will optimize clz(1 << i) to bitsize - i-1

    for (u32 i = 0; i < 32; i++) {
        REQUIRE_EQUAL(sham::log2_pow2_num<u32>(bitshift_noinline(1_u32, i)), i);
    }

    for (u64 i = 0; i < 64; i++) {
        REQUIRE_EQUAL(sham::log2_pow2_num<u64>(bitshift_noinline(1_u64, i)), i);
    }
}

TestStart(Unittest, "shambackends/math.hpp:max_component", shambackendsmathmax_component, 1) {

    sycl::vec<f32, 2> a = {1._f32, 2._f32};
    sycl::vec<f32, 3> b = {1._f32, 2._f32, 3._f32};
    sycl::vec<f32, 4> c = {1._f32, 2._f32, 3._f32, 4._f32};
    sycl::vec<f32, 8> d = {1._f32, 2._f32, 3._f32, 4._f32, 5._f32, 6._f32, 7._f32, 8._f32};
    sycl::vec<f32, 16> e
        = {1._f32,
           2._f32,
           3._f32,
           4._f32,
           5._f32,
           6._f32,
           7._f32,
           8._f32,
           9._f32,
           10._f32,
           11._f32,
           12._f32,
           13._f32,
           14._f32,
           15._f32,
           16._f32};

    REQUIRE_EQUAL(sham::max_component(a), 2._f32);
    REQUIRE_EQUAL(sham::max_component(b), 3._f32);
    REQUIRE_EQUAL(sham::max_component(c), 4._f32);
    REQUIRE_EQUAL(sham::max_component(d), 8._f32);
    REQUIRE_EQUAL(sham::max_component(e), 16._f32);
}
