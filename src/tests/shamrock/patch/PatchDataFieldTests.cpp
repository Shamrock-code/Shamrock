// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/StlContainerConversion.hpp"

#include "shamalgs/serialize.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

#include <set>

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_buf", testpatchdatafieldserialize, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser;

    ser.allocate(field.serialize_buf_byte_size());
    field.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_buf(ser2, name, nvar);

        shamtest::asserts().assert_bool("input match out", field.check_field_match(buf2));
    }
}

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_full", testpatchdatafieldserializefull, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser;

    ser.allocate(field.serialize_full_byte_size());
    field.serialize_full(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_full(ser2);

        shamtest::asserts().assert_bool("input match out", field.check_field_match(buf2));
    }
}

TestStart(Unittest, "shamrock/patch/PatchDataField::get_ids_..._where", testgetelemwithrange, 1) {

    u32 len                   = 10000;
    u32 nvar                  = 1;
    std::string name          = "testfield";
    f64 vmin                  = 0;
    f64 vmax                  = 1000;
    PatchDataField<f64> field = PatchDataField<f64>::mock_field(0x111, len, name, nvar, 0, 2000);

    std::set<u32> idx_cd = field.get_ids_set_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", idx_cd.size());

    std::vector<u32> idx_cd_vec = field.get_ids_vec_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", idx_cd_vec.size());

    auto idx_cd_sycl = field.get_ids_buf_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", std::get<1>(idx_cd_sycl));

    // compare content
    _Assert(bool(std::get<0>(idx_cd_sycl)))

    if (std::get<0>(idx_cd_sycl)) {
        _Assert(idx_cd == shambase::set_from_vector(idx_cd_vec)) 
        _Assert(
            idx_cd == shambase::set_from_vector(
                shamalgs::memory::buf_to_vec(
                    *std::get<0>(idx_cd_sycl), std::get<1>(idx_cd_sycl)
                    )
                )
        )
    }
}
