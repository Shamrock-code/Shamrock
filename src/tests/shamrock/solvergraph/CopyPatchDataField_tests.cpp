// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>

TestStart(Unittest, "shamrock/solvergraph/CopyPatchDataField", testCopyPatchDataFieldBasic, 1) {
    using namespace shamrock::solvergraph;
    using namespace shamrock::patch;

    {
        // Test field copying with multiple patches
        auto original_field_refs = FieldRefs<f64>::make_shared("original_multi", "orig_multi");
        auto target_field        = std::make_shared<Field<f64>>(1, "target_multi", "tgt_multi");

        auto copy_node = std::make_shared<CopyPatchDataField<f64>>();
        copy_node->set_edges(original_field_refs, target_field);

        u64 seed1      = 0x9ABC;
        u64 seed2      = 0xDEF0;
        u32 obj_count1 = 75;
        u32 obj_count2 = 25;

        auto mock_field1
            = PatchDataField<f64>::mock_field(seed1, obj_count1, "test_f64_field_1", 1);
        auto mock_field2
            = PatchDataField<f64>::mock_field(seed2, obj_count2, "test_f64_field_2", 1);

        original_field_refs->get_refs().add_obj(10, std::ref(mock_field1));
        original_field_refs->get_refs().add_obj(20, std::ref(mock_field2));

        copy_node->evaluate();

        REQUIRE_EQUAL(target_field->get_refs().get_element_count(), 2);
        REQUIRE_EQUAL(target_field->get_field(10).get_obj_cnt(), obj_count1);
        REQUIRE_EQUAL(target_field->get_field(20).get_obj_cnt(), obj_count2);

        auto original_data1 = mock_field1.get_buf().copy_to_stdvec();
        auto target_data1   = target_field->get_field(10).get_buf().copy_to_stdvec();
        REQUIRE_EQUAL(original_data1, target_data1);

        auto original_data2 = mock_field2.get_buf().copy_to_stdvec();
        auto target_data2   = target_field->get_field(20).get_buf().copy_to_stdvec();
        REQUIRE_EQUAL(original_data2, target_data2);
    }
}
