// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

inline void
test_serialize_basic(std::shared_ptr<PatchDataLayerLayout> pdl_ptr, PatchDataLayer &pdat) {

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
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

    { // case where the total data count exceed the int 32 limit

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
    }
}
