// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/logs/loglevels.hpp"
#include "shamalgs/serialize.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

using namespace shamrock::patch;

inline void test_serialize_basic(
    std::shared_ptr<PatchDataLayerLayout> pdl_ptr, PatchDataLayer &pdat) {

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    if (pdat.serialize_buf_byte_size().get_total_size() > std::numeric_limits<u32>::max()) {
        REQUIRE_EXCEPTION_THROW(ser.allocate(pdat.serialize_buf_byte_size()), std::runtime_error);
        return;
    }

    ser.allocate(pdat.serialize_buf_byte_size());

    pdat.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchDataLayer pdat2 = PatchDataLayer::deserialize_buf(ser2, pdl_ptr);

        REQUIRE_NAMED("input match out", pdat == pdat2);
    }
}

TestStart(
    Unittest, "shamrock/patch/PatchDataLayer::serialize_buf", testpatchdatalayerserialize, 1) {
    using namespace shamrock::patch;

    { // basic case

        u32 obj = 1000;

        std::shared_ptr<PatchDataLayerLayout> pdl_ptr = std::make_shared<PatchDataLayerLayout>();
        auto &pdl                                     = *pdl_ptr;

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

        PatchDataLayer pdat = PatchDataLayer::mock_patchdata(0x111, obj, pdl_ptr);

        test_serialize_basic(pdl_ptr, pdat);
    }

    u64 dev_mem_size
        = shamsys::instance::get_compute_scheduler().get_queue().get_device_prop().global_mem_size;

    if (dev_mem_size > 4'000'000'000) { // case where the total data count exceed the int 32 limit

        u32 obj = 1e7;

        std::shared_ptr<PatchDataLayerLayout> pdl_ptr = std::make_shared<PatchDataLayerLayout>();
        auto &pdl                                     = *pdl_ptr;

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

        // weirdness to avoid having to generate that many objects in CI
        PatchDataLayer pdat = PatchDataLayer::mock_patchdata(0x111, 1000, pdl_ptr);
        pdat.resize(obj);

        shamlog_info_ln("test", pdat.serialize_buf_byte_size().get_total_size());

        REQUIRE_NAMED(
            "total size exceed int 32 limit",
            pdat.serialize_buf_byte_size().get_total_size() > std::numeric_limits<i32>::max());

        test_serialize_basic(pdl_ptr, pdat);
    } else {
        REQUIRE_NAMED("dev_mem_size < 4'000'000'000, skipping test", false);
    }
}

TestStart(
    Unittest,
    "shamrock/patch/PatchDataLayer::extract_elements",
    testpatchdatalayerextractelements,
    1) {
    using namespace shamrock::patch;

    { // Test extract_elements with multi-variable fields
        std::shared_ptr<PatchDataLayerLayout> pdl_ptr = std::make_shared<PatchDataLayerLayout>();
        auto &pdl                                     = *pdl_ptr;

        pdl.add_field<f32_3>("position", 1);
        pdl.add_field<f64_4>("velocity", 2);

        u32 obj_cnt                = 8;
        PatchDataLayer pdat_source = PatchDataLayer::mock_patchdata(0x456, obj_cnt, pdl_ptr);

        std::vector<u32> indices_vec = {0, 2, 4, 6};
        auto sched                   = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceBuffer<u32> indices(indices_vec.size(), sched);
        indices.copy_from_stdvec(indices_vec);

        auto &pos_field = pdat_source.get_field<f32_3>(pdl.get_field_idx<f32_3>("position"));
        auto &vel_field = pdat_source.get_field<f64_4>(pdl.get_field_idx<f64_4>("velocity"));

        std::vector<f32_3> original_pos = pos_field.copy_to_stdvec();
        std::vector<f64_4> original_vel = vel_field.copy_to_stdvec();

        PatchDataLayer pdat_dest{pdl_ptr};
        pdat_source.extract_elements(indices, pdat_dest);

        REQUIRE_EQUAL(pdat_dest.get_obj_cnt(), 4);
        REQUIRE_EQUAL(pdat_source.get_obj_cnt(), 4);

        auto &pos_dest = pdat_dest.get_field<f32_3>(pdl.get_field_idx<f32_3>("position"));
        auto &vel_dest = pdat_dest.get_field<f64_4>(pdl.get_field_idx<f64_4>("velocity"));

        std::vector<f32_3> extracted_pos = pos_dest.copy_to_stdvec();
        std::vector<f64_4> extracted_vel = vel_dest.copy_to_stdvec();

        std::vector<f32_3> expected_pos
            = {original_pos[0], original_pos[2], original_pos[4], original_pos[6]};
        std::vector<f64_4> expected_vel;
        const u32 nvar_vel = vel_field.get_nvar();
        for (const u32 obj_idx : indices_vec) {
            for (u32 var_idx = 0; var_idx < nvar_vel; ++var_idx) {
                expected_vel.push_back(original_vel[obj_idx * nvar_vel + var_idx]);
            }
        }

        REQUIRE_EQUAL_CUSTOM_COMP(extracted_pos, expected_pos, sham::equals);
        REQUIRE_EQUAL_CUSTOM_COMP(extracted_vel, expected_vel, sham::equals);
    }

    { // Test extract_elements with empty indices
        std::shared_ptr<PatchDataLayerLayout> pdl_ptr = std::make_shared<PatchDataLayerLayout>();
        auto &pdl                                     = *pdl_ptr;

        pdl.add_field<f32>("field", 1);

        u32 obj_cnt                = 5;
        PatchDataLayer pdat_source = PatchDataLayer::mock_patchdata(0x789, obj_cnt, pdl_ptr);

        auto sched = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceBuffer<u32> indices(0, sched);

        PatchDataLayer pdat_dest{pdl_ptr};
        pdat_source.extract_elements(indices, pdat_dest);

        REQUIRE_EQUAL(pdat_dest.get_obj_cnt(), 0);
        REQUIRE_EQUAL(pdat_dest.is_empty(), true);
        REQUIRE_EQUAL(pdat_source.get_obj_cnt(), 5);
        REQUIRE_EQUAL(pdat_source.is_empty(), false);
    }
}
