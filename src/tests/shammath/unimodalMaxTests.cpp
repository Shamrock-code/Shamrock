// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/unimodal_max.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>

NEW_TEST(Unittest, "shammath/unimodal_max_value", 1) {

    // interior max
    f64 max_parabola = shammath::unimodal_max_value<f64>(
        [](f64 x) {
            return 2.0 - (x - 0.3) * (x - 0.3);
        },
        -1.0,
        4.0);
    REQUIRE_FLOAT_EQUAL(max_parabola, 2.0, 1e-8);

    // monotonic functions: the max sits on a boundary
    f64 max_increasing = shammath::unimodal_max_value<f64>(
        [](f64 x) {
            return std::exp(x);
        },
        0.0,
        1.0);
    REQUIRE_FLOAT_EQUAL(max_increasing, std::exp(1.0), 1e-12);

    f64 max_decreasing = shammath::unimodal_max_value<f64>(
        [](f64 x) {
            return 1.0 / x;
        },
        0.5,
        10.0);
    REQUIRE_FLOAT_EQUAL(max_decreasing, 2.0, 1e-12);

    // r * sigma(r) of a tapered disc, sigma = r^-1.5 (1 - sqrt(r_in / r)): max at r = 4 r_in
    f64 r_in  = 1.0;
    auto f_rs = [&](f64 r) {
        return r * std::pow(r, -1.5) * (1.0 - std::sqrt(r_in / r));
    };
    f64 max_rs = shammath::unimodal_max_value<f64>(f_rs, r_in, 100.0);
    REQUIRE_FLOAT_EQUAL(max_rs / f_rs(4.0 * r_in), 1.0, 1e-7);
}
