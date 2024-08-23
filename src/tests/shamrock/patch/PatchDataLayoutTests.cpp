// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamrock/patch/PatchDataLayout::serialize_json", testpdlserjson, 1) {
    using namespace shamrock::patch;

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

    nlohmann::json j = pdl;

    // logger::raw_ln(j.dump(4));

    PatchDataLayout pdl_out = j.get<PatchDataLayout>();

    REQUIRE(pdl == pdl_out);
}
