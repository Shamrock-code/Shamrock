// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/PatchDataLayout::serialize_json", testpdlserjson, 1) {
    using namespace shamrock::patch;

    PatchDataLayout pdl = PatchDataLayout(std::vector<std::string>{"layer1", "layer2", "layer3"});

    auto &layer1 = pdl.get_layer_ref(0);
    auto &layer2 = pdl.get_layer_ref(1);
    auto &layer3 = pdl.get_layer_ref(2);

    layer1.add_field<f32>("f32", 1);
    layer1.add_field<f32_2>("f32_2", 1);
    layer1.add_field<f32_3>("f32_3", 1);
    layer1.add_field<f32_3>("f32_3'", 1);

    layer2.add_field<f32_3>("f32_3''", 1);
    layer2.add_field<f32_4>("f32_4", 1);
    layer2.add_field<f32_8>("f32_8", 1);
    layer2.add_field<f32_16>("f32_16", 1);
    layer2.add_field<f64>("f64", 1);

    layer3.add_field<f64_2>("f64_2", 1);
    layer3.add_field<f64_3>("f64_3", 1);
    layer3.add_field<f64_4>("f64_4", 2);
    layer3.add_field<f64_8>("f64_8", 1);
    layer3.add_field<f64_16>("f64_16", 1);
    layer3.add_field<u32>("u32", 1);
    layer3.add_field<u64>("u64", 1);

    nlohmann::json j = pdl;

    logger::raw_ln(j.dump(4));

    PatchDataLayout pdl_out = j.get<PatchDataLayout>();

    REQUIRE(pdl == pdl_out);
}

// this test is necessary to ensure backward compatibility before layers were introduced
TestStart(
    Unittest,
    "shamrock/patch/PatchDataLayout::deserialize_json_from_layer",
    testpdlserjsonfromlayer,
    1) {
    using namespace shamrock::patch;

    PatchDataLayerLayout pdl_layer;
    pdl_layer.add_field<f32>("f32", 1);
    pdl_layer.add_field<f32_2>("f32_2", 1);
    pdl_layer.add_field<f32_3>("f32_3", 1);
    pdl_layer.add_field<f32_3>("f32_3'", 1);

    pdl_layer.add_field<f32_3>("f32_3''", 1);
    pdl_layer.add_field<f32_4>("f32_4", 1);
    pdl_layer.add_field<f32_8>("f32_8", 1);
    pdl_layer.add_field<f32_16>("f32_16", 1);
    pdl_layer.add_field<f64>("f64", 1);

    nlohmann::json j = pdl_layer;

    logger::raw_ln(j.dump(4));

    PatchDataLayout pdl_layer_out = j.get<PatchDataLayout>();

    PatchDataLayout pdl_ref = PatchDataLayout();
    pdl_ref.layer_layouts.push_back(
        PatchDataLayout::LayerEntry{std::make_shared<PatchDataLayerLayout>(pdl_layer), "main"});

    REQUIRE(pdl_ref == pdl_layer_out);
}
