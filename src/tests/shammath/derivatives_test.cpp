// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammath/derivatives.hpp"
#include "shamtest/shamtest.hpp"
#include <cmath>

TestStart(Unittest, "shammath/derivatives/derivative_upwind", test_derivative_upwind, 1) {

    using namespace shammath;

    f64 x = 1.0;

    auto f = [](f64 x) {
        return std::exp(x);
    };
    auto df = [](f64 x) {
        return std::exp(x);
    };

    f64 val = derivative_upwind<f64>(x, estim_deriv_step<f64>(2), [&](f64 x) {
        return f(x);
    });

    REQUIRE_FLOAT_EQUAL(df(x), val, 8.3e-06);
}

TestStart(Unittest, "shammath/derivatives/derivative_centered", test_derivative_centered, 1) {

    using namespace shammath;

    f64 x = 1.0;

    auto f = [](f64 x) {
        return std::exp(x);
    };
    auto df = [](f64 x) {
        return std::exp(x);
    };

    f64 val = derivative_centered<f64>(x, estim_deriv_step<f64>(3), [&](f64 x) {
        return f(x);
    });

    REQUIRE_FLOAT_EQUAL(df(x), val, 6.8e-09);
}

TestStart(
    Unittest, "shammath/derivatives/derivative_3point_forward", test_derivative_3point_forward, 1) {

    using namespace shammath;

    f64 x = 1.0;

    auto f = [](f64 x) {
        return std::exp(x);
    };
    auto df = [](f64 x) {
        return std::exp(x);
    };

    f64 val = derivative_3point_forward<f64>(x, estim_deriv_step<f64>(3), [&](f64 x) {
        return f(x);
    });

    REQUIRE_FLOAT_EQUAL(df(x), val, 1.4e-08);
}

TestStart(
    Unittest,
    "shammath/derivatives/derivative_3point_backward",
    test_derivative_3point_backward,
    1) {

    using namespace shammath;

    f64 x = 1.0;

    auto f = [](f64 x) {
        return std::exp(x);
    };
    auto df = [](f64 x) {
        return std::exp(x);
    };

    f64 val = derivative_3point_backward<f64>(x, estim_deriv_step<f64>(3), [&](f64 x) {
        return f(x);
    });

    REQUIRE_FLOAT_EQUAL(df(x), val, 1.4e-08);
}

TestStart(
    Unittest,
    "shammath/derivatives/derivative_5point_midpoint",
    test_derivative_5point_midpoint,
    1) {

    using namespace shammath;

    f64 x = 1.0;

    auto f = [](f64 x) {
        return std::exp(x);
    };
    auto df = [](f64 x) {
        return std::exp(x);
    };

    f64 val = derivative_5point_midpoint<f64>(x, estim_deriv_step<f64>(4), [&](f64 x) {
        return f(x);
    });

    REQUIRE_FLOAT_EQUAL(df(x), val, 3.1e-13);
}
