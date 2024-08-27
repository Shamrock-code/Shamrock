// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/serialize.hpp"
#include "shamrock/patch/PatchData.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/PatchData::serialize_buf", testpatchdataserialize, 1) {
    using namespace shamrock::patch;

    u32 obj = 1000;

    PatchDataLayout pdl;

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

    PatchData pdat = PatchData::mock_patchdata(0x111, obj, pdl);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());
    ser.allocate(pdat.serialize_buf_byte_size());
    pdat.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchData pdat2 = PatchData::deserialize_buf(ser2, pdl);

        shamtest::asserts().assert_bool("input match out", pdat == pdat2);
    }
}
