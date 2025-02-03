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
}
