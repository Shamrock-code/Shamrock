// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

    shamtest::asserts().assert_equal("v1", sham::has_nan(v1), false);
    shamtest::asserts().assert_equal("v2", sham::has_nan(v2), true);
    shamtest::asserts().assert_equal("v3", sham::has_nan(v3), false);
}

TestStart(Unittest, "sham::has_inf", testshambasehasinf, 1) {

    f32_3 v1{0, 0, 0};
    f32_3 v2{std::nan(""), 0, 0};
    f32_3 v3{shambase::VectorProperties<f32>::get_inf(), 0, 0};

    shamtest::asserts().assert_equal("v1", sham::has_inf(v1), false);
    shamtest::asserts().assert_equal("v2", sham::has_inf(v2), false);
    shamtest::asserts().assert_equal("v3", sham::has_inf(v3), true);
}

TestStart(Unittest, "sham::has_nan_or_inf", testshambasehasnaninf, 1) {

    f32_3 v1{0, 0, 0};
    f32_3 v2{std::nan(""), 0, 0};
    f32_3 v3{shambase::VectorProperties<f32>::get_inf(), 0, 0};

    shamtest::asserts().assert_equal("v1", sham::has_nan_or_inf(v1), false);
    shamtest::asserts().assert_equal("v2", sham::has_nan_or_inf(v2), true);
    shamtest::asserts().assert_equal("v3", sham::has_nan_or_inf(v3), true);
}
