// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/legacy/patch/utility/patch_field.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/legacy/sycl_mpi_interop.hpp"
#include "shamtest/shamtest.hpp"
#include <random>

const std::string base = "core/patch/base/patchdata_field";

/**
 *   Checking PatchDataField status after constructor
 */
TestStart(Unittest, base + ":constructor", patch_data_field_constructor, 1) {

    PatchDataField<f32> d_check("test", 1);

    REQUIRE_NAMED("name field match", d_check.get_name() == "test");
    REQUIRE_NAMED("nvar field match", d_check.get_nvar() == 1);
    REQUIRE_NAMED("buffer allocated but empty", d_check.get_buf().is_empty());

    REQUIRE_NAMED("is new field empty", d_check.get_val_cnt() == 0);
    REQUIRE_NAMED("is new field empty", d_check.get_obj_cnt() == 0);
}

TestStart(Unittest, base + ":patch_data_field_check_match", patch_data_field_check_match, 1) {
    std::mt19937 eng(0x1111);

    PatchDataField<f32> d_check("test", 1);
    d_check.gen_mock_data(10000, eng);

    REQUIRE_NAMED("reflexivity", d_check.check_field_match(d_check));
}
