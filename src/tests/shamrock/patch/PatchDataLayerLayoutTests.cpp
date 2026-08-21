// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/legacy/patch/base/enabled_fields.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamtest/shamtest.hpp"
#include <nlohmann/json.hpp>

NEW_TEST(Unittest, "shamrock/patch/PatchDataLayerLayout::serialize_json", 1) {
    using namespace shamrock::patch;

    PatchDataLayerLayout pdl;

    pdl.add_field<f32>("f32", 1);
    pdl.add_field<f32_2>("f32_2", 1);

    pdl.add_field<f32_3>("f32_3", 1);
    pdl.add_field<f32_3>("f32_3'", 1);
    pdl.add_field<f32_3>("f32_3''", 1);

    pdl.add_field<f32_4>("f32_4", 1);
    pdl.add_field<f32_8>("f32_8", 1);
    pdl.add_field<f32_16>("f32_16", 1);
    pdl.add_field<f64>("f64", 1);
    pdl.add_field<f64_2>("f64_2", 1);
    pdl.add_field<f64_3>("f64_3", 1);
    pdl.add_field<f64_4>("f64_4", 2);
    pdl.add_field<f64_8>("f64_8", 1);
    pdl.add_field<f64_16>("f64_16", 1);

    pdl.add_field<u32>("u32", 1);
    pdl.add_field<u64>("u64", 1);

    nlohmann::json j = pdl;

    // logger::raw_ln(j.dump(4));

    PatchDataLayerLayout pdl_out = j.get<PatchDataLayerLayout>();

    REQUIRE(pdl == pdl_out);
}

NEW_TEST(Unittest, "shamrock/patch/PatchDataLayerLayout::add_field", 1) {
    using namespace shamrock::patch;

    PatchDataLayerLayout pdl;

    u32 nfields = 0;
#define X(type)                                                                                    \
    pdl.add_field<type>(#type, 1);                                                                 \
    nfields++;
    XMAC_LIST_ENABLED_FIELD
#undef X

    REQUIRE_EQUAL(pdl.get_field_names().size(), nfields);

    REQUIRE_EXCEPTION_THROW(pdl.add_field<f32>("f32", 1), std::invalid_argument);
}
