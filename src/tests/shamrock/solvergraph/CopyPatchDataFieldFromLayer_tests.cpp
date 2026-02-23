// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/CopyPatchDataFieldFromLayer.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>

template<typename T>
void verify_copy(
    shamrock::solvergraph::Field<T> &target_field,
    const shamrock::patch::PatchDataLayer &source_patchdata,
    u32 field_idx,
    u32 obj_count,
    u32 expected_nvar) {

    REQUIRE_EQUAL(target_field.get_refs().get_element_count(), 1);
    auto &out_field = target_field.get_refs().get(1).get();
    REQUIRE_EQUAL(out_field.get_obj_cnt(), obj_count);
    REQUIRE_EQUAL(out_field.get_nvar(), expected_nvar);

    auto &source_field = source_patchdata.get_field<T>(field_idx);
    auto source_data   = source_field.get_buf().copy_to_stdvec();
    auto target_data   = out_field.get_buf().copy_to_stdvec();
    REQUIRE_EQUAL_CUSTOM_COMP(source_data, target_data, sham::equals);
}

TestStart(
    Unittest,
    "shamrock/solvergraph/CopyPatchDataFieldFromLayer",
    testCopyPatchDataFieldFromLayerMultipleFields,
    1) {
    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    // Create a layout with multiple fields, some with nvar > 1
    auto layout = std::make_shared<PatchDataLayerLayout>();
    layout->add_field<f32>("scalar_field", 1);     // nvar = 1
    layout->add_field<f32_3>("vector_field", 3);   // nvar = 3
    layout->add_field<f64>("double_field", 1);     // nvar = 1
    layout->add_field<f32_3>("velocity_field", 2); // nvar = 2
    layout->add_field<u64>("index_field", 1);      // nvar = 1

    // Create mock patch data with multiple objects
    u64 seed              = 0xABCD;
    u32 obj_count         = 250;
    auto source_patchdata = PatchDataLayer::mock_patchdata(seed, obj_count, layout);

    // Create source data edge (PatchDataLayerRefs)
    auto source_refs = std::make_shared<PatchDataLayerRefs>("source", "source_refs");
    source_refs->patchdatas.add_obj(1, std::ref(source_patchdata));

    // Test copying vector_field (nvar = 3)
    {
        std::string field_name = "vector_field";
        u32 field_idx          = layout->get_field_idx<f32_3>(field_name);
        u32 expected_nvar      = 3;

        // Create the copy node
        auto copy_node = std::make_shared<CopyPatchDataFieldFromLayer<f32_3>>(field_idx);

        // Create target field edge
        auto target_field = std::make_shared<Field<f32_3>>(3, "target_field", "target_field");

        // Set edges
        copy_node->set_edges(source_refs, target_field);

        // Execute the copy operation
        copy_node->evaluate();

        // Verify the copy
        verify_copy<f32_3>(*target_field, source_patchdata, field_idx, obj_count, expected_nvar);
    }

    // Test copying velocity_field (nvar = 2)
    {
        std::string field_name = "velocity_field";
        u32 field_idx          = layout->get_field_idx<f32_3>(field_name);
        u32 expected_nvar      = 2;

        // Create the copy node using the constructor that takes layout and field name
        auto copy_node = std::make_shared<CopyPatchDataFieldFromLayer<f32_3>>(layout, field_name);

        // Create target field edge
        auto target_field = std::make_shared<Field<f32_3>>(2, "target_velocity", "target_velocity");

        // Set edges
        copy_node->set_edges(source_refs, target_field);

        // Execute the copy operation
        copy_node->evaluate();

        // Verify the copy
        verify_copy<f32_3>(*target_field, source_patchdata, field_idx, obj_count, expected_nvar);
    }

    // Test copying scalar_field (nvar = 1)
    {
        std::string field_name = "scalar_field";
        u32 field_idx          = layout->get_field_idx<f32>(field_name);
        u32 expected_nvar      = 1;

        // Create the copy node
        auto copy_node = std::make_shared<CopyPatchDataFieldFromLayer<f32>>(field_idx);

        // Create target field edge
        auto target_field = std::make_shared<Field<f32>>(1, "target_scalar", "target_scalar");

        // Set edges
        copy_node->set_edges(source_refs, target_field);

        // Execute the copy operation
        copy_node->evaluate();

        // Verify the copy
        verify_copy<f32>(*target_field, source_patchdata, field_idx, obj_count, expected_nvar);
    }
}
