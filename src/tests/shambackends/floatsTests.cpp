// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/floats.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "sham::has_nan", testshambasehasnan, 1) {

    f32_3 v1{0, 0, 0};
    f32_3 v2{std::nan(""), 0, 0};
    f32_3 v3{shambase::VectorProperties<f32>::get_inf(), 0, 0};

    REQUIRE_EQUAL_NAMED("v1", sham::has_nan(v1), false);
    REQUIRE_EQUAL_NAMED("v2", sham::has_nan(v2), true);
    REQUIRE_EQUAL_NAMED("v3", sham::has_nan(v3), false);
}

TestStart(Unittest, "sham::has_inf", testshambasehasinf, 1) {

    f32_3 v1{0, 0, 0};
    f32_3 v2{std::nan(""), 0, 0};
    f32_3 v3{shambase::VectorProperties<f32>::get_inf(), 0, 0};

    REQUIRE_EQUAL_NAMED("v1", sham::has_inf(v1), false);
    REQUIRE_EQUAL_NAMED("v2", sham::has_inf(v2), false);
    REQUIRE_EQUAL_NAMED("v3", sham::has_inf(v3), true);
}

TestStart(Unittest, "sham::has_nan_or_inf", testshambasehasnaninf, 1) {

    f32_3 v1{0, 0, 0};
    f32_3 v2{std::nan(""), 0, 0};
    f32_3 v3{shambase::VectorProperties<f32>::get_inf(), 0, 0};

    REQUIRE_EQUAL_NAMED("v1", sham::has_nan_or_inf(v1), false);
    REQUIRE_EQUAL_NAMED("v2", sham::has_nan_or_inf(v2), true);
    REQUIRE_EQUAL_NAMED("v3", sham::has_nan_or_inf(v3), true);
}
