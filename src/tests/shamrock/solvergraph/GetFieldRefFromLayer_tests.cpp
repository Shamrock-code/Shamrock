// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <random>
#include <stdexcept>

TestStart(Unittest, "shamrock/solvergraph/GetFieldRefFromLayer", testGetFieldRefFromLayer, 1) {
    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    // Create a layout with various field types
    auto layout = std::make_shared<PatchDataLayerLayout>();
    layout->add_field<f32>("scalar_field", 1);
    layout->add_field<f32_3>("vector_field", 2); // 2 variables per object
    layout->add_field<u64>("index_field", 1);
    layout->add_field<f64>("double_field", 1);

    // Create mock patch data with multiple objects
    u64 seed              = 0xABCD;
    u32 obj_count         = 250;
    auto source_patchdata = PatchDataLayer::mock_patchdata(seed, obj_count, layout);

    // Create source data edge
    auto source_refs = std::make_shared<PatchDataLayerRefs>("source", "source_refs");
    source_refs->patchdatas.add_obj(1, std::ref(source_patchdata));

    auto compare_data = [](auto &a, auto &b) {
        bool ret = a.size() == b.size();
        for (u32 i = 0; i < a.size(); i++) {
            ret = ret && (sham::equals(a[i], b[i]));
        }
        return ret;
    };

    // first field
    {
        using T                = f32;
        std::string field_name = "scalar_field";
        u32 field_idx          = layout->get_field_idx<T>(field_name);

        // Create output field refs edge
        auto out_field_refs = std::make_shared<FieldRefs<T>>("out_field", "out_field");

        // Create the GetFieldRefFromLayer node for the vector field
        auto get_field_node = std::make_shared<GetFieldRefFromLayer<T>>(field_idx);
        get_field_node->set_edges(source_refs, out_field_refs);

        // Execute the field extraction
        get_field_node->evaluate();

        // Verify output field refs were created correctly
        REQUIRE_EQUAL(out_field_refs->get_refs().get_element_count(), 1);

        auto &out_field = out_field_refs->get_refs().get(1).get();

        REQUIRE_EQUAL(out_field.get_obj_cnt(), obj_count);
        REQUIRE_EQUAL(out_field.get_name(), field_name);
        REQUIRE_EQUAL(out_field.get_nvar(), 1);

        // Verify the extracted field is actually the same as the source field
        auto &source_pdat  = source_refs->patchdatas.get(1).get();
        auto &source_field = source_pdat.get_field<T>(field_idx);

        REQUIRE_EQUAL_CUSTOM_COMP(
            source_field.get_buf().copy_to_stdvec(),
            out_field.get_buf().copy_to_stdvec(),
            compare_data);
    }

    // second field
    {
        using T                = f32_3;
        std::string field_name = "vector_field";
        u32 field_idx          = layout->get_field_idx<T>(field_name);

        // Create output field refs edge
        auto out_field_refs = std::make_shared<FieldRefs<T>>("out_field", "out_field");

        // Create the GetFieldRefFromLayer node for the vector field
        auto get_field_node = std::make_shared<GetFieldRefFromLayer<T>>(field_idx);
        get_field_node->set_edges(source_refs, out_field_refs);

        // Execute the field extraction
        get_field_node->evaluate();

        // Verify output field refs were created correctly
        REQUIRE_EQUAL(out_field_refs->get_refs().get_element_count(), 1);

        auto &out_field = out_field_refs->get_refs().get(1).get();

        REQUIRE_EQUAL(out_field.get_obj_cnt(), obj_count);
        REQUIRE_EQUAL(out_field.get_name(), field_name);
        REQUIRE_EQUAL(out_field.get_nvar(), 2);

        // Verify the extracted field is actually the same as the source field
        auto &source_pdat  = source_refs->patchdatas.get(1).get();
        auto &source_field = source_pdat.get_field<T>(field_idx);

        REQUIRE_EQUAL_CUSTOM_COMP(
            source_field.get_buf().copy_to_stdvec(),
            out_field.get_buf().copy_to_stdvec(),
            compare_data);
    }

    // third field
    {
        using T                = u64;
        std::string field_name = "index_field";
        u32 field_idx          = layout->get_field_idx<T>(field_name);

        // Create output field refs edge
        auto out_field_refs = std::make_shared<FieldRefs<T>>("out_field", "out_field");

        // Create the GetFieldRefFromLayer node for the vector field
        auto get_field_node = std::make_shared<GetFieldRefFromLayer<T>>(field_idx);
        get_field_node->set_edges(source_refs, out_field_refs);

        // Execute the field extraction
        get_field_node->evaluate();

        // Verify output field refs were created correctly
        REQUIRE_EQUAL(out_field_refs->get_refs().get_element_count(), 1);

        auto &out_field = out_field_refs->get_refs().get(1).get();

        REQUIRE_EQUAL(out_field.get_obj_cnt(), obj_count);
        REQUIRE_EQUAL(out_field.get_name(), field_name);
        REQUIRE_EQUAL(out_field.get_nvar(), 1);

        // Verify the extracted field is actually the same as the source field
        auto &source_pdat  = source_refs->patchdatas.get(1).get();
        auto &source_field = source_pdat.get_field<T>(field_idx);

        REQUIRE_EQUAL_CUSTOM_COMP(
            source_field.get_buf().copy_to_stdvec(),
            out_field.get_buf().copy_to_stdvec(),
            compare_data);
    }

    // fourth field
    {
        using T                = f64;
        std::string field_name = "double_field";
        u32 field_idx          = layout->get_field_idx<T>(field_name);

        // Create output field refs edge
        auto out_field_refs = std::make_shared<FieldRefs<T>>("out_field", "out_field");

        // Create the GetFieldRefFromLayer node for the vector field
        auto get_field_node = std::make_shared<GetFieldRefFromLayer<T>>(field_idx);
        get_field_node->set_edges(source_refs, out_field_refs);

        // Execute the field extraction
        get_field_node->evaluate();

        // Verify output field refs were created correctly
        REQUIRE_EQUAL(out_field_refs->get_refs().get_element_count(), 1);

        auto &out_field = out_field_refs->get_refs().get(1).get();

        REQUIRE_EQUAL(out_field.get_obj_cnt(), obj_count);
        REQUIRE_EQUAL(out_field.get_name(), field_name);
        REQUIRE_EQUAL(out_field.get_nvar(), 1);

        // Verify the extracted field is actually the same as the source field
        auto &source_pdat  = source_refs->patchdatas.get(1).get();
        auto &source_field = source_pdat.get_field<T>(field_idx);

        REQUIRE_EQUAL_CUSTOM_COMP(
            source_field.get_buf().copy_to_stdvec(),
            out_field.get_buf().copy_to_stdvec(),
            compare_data);
    }
}
