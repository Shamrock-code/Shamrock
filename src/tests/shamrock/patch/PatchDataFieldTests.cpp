// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/StlContainerConversion.hpp"
#include "shamalgs/serialize.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <set>
#include <vector>

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_buf", testpatchdatafieldserialize, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    ser.allocate(field.serialize_buf_byte_size());
    field.serialize_buf(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_buf(ser2, name, nvar);

        REQUIRE_NAMED("input match out", field.check_field_match(buf2));
    }
}

TestStart(
    Unittest, "shamrock/patch/PatchDataField::serialize_full", testpatchdatafieldserializefull, 1) {

    u32 len                     = 1000;
    u32 nvar                    = 2;
    std::string name            = "testfield";
    PatchDataField<u32_3> field = PatchDataField<u32_3>::mock_field(0x111, len, name, nvar);

    shamalgs::SerializeHelper ser(shamsys::instance::get_compute_scheduler_ptr());

    ser.allocate(field.serialize_full_byte_size());
    field.serialize_full(ser);

    auto recov = ser.finalize();

    {
        shamalgs::SerializeHelper ser2(
            shamsys::instance::get_compute_scheduler_ptr(), std::move(recov));

        PatchDataField<u32_3> buf2 = PatchDataField<u32_3>::deserialize_full(ser2);

        REQUIRE_NAMED("input match out", field.check_field_match(buf2));
    }
}

inline void check_pdat_get_ids_where(u32 len, u32 nvar, std::string name, f64 vmin, f64 vmax) {

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

    auto idx_cd_shambuf = field.get_ids_where(
        [](auto access, u32 id, f64 vmin, f64 vmax) {
            f64 tmp = access[id];
            return tmp > vmin && tmp < vmax;
        },
        vmin,
        vmax);

    logger::raw_ln("found : ", idx_cd_shambuf.get_size());

    // compare content
    REQUIRE(bool(std::get<0>(idx_cd_sycl)) == (idx_cd.size() != 0));

    if (std::get<0>(idx_cd_sycl)) {
        REQUIRE(idx_cd == shambase::set_from_vector(idx_cd_vec));
        REQUIRE(
            idx_cd
            == shambase::set_from_vector(
                shamalgs::memory::buf_to_vec(*std::get<0>(idx_cd_sycl), std::get<1>(idx_cd_sycl))));
        REQUIRE(idx_cd == shambase::set_from_vector(idx_cd_shambuf.copy_to_stdvec()));
    }
}

TestStart(Unittest, "shamrock/patch/PatchDataField::get_ids_..._where", testgetelemwithrange, 1) {

    {
        u32 len          = 10000;
        u32 nvar         = 1;
        std::string name = "testfield";
        f64 vmin         = 0;
        f64 vmax         = 1000;

        check_pdat_get_ids_where(len, nvar, name, vmin, vmax);
    }
    {
        u32 len          = 0;
        u32 nvar         = 1;
        std::string name = "testfield";
        f64 vmin         = 0;
        f64 vmax         = 1000;

        check_pdat_get_ids_where(len, nvar, name, vmin, vmax);
    }
}

TestStart(Unittest, "shamrock/patch/PatchDataField::remove_ids", testpdatremoveids, 1) {

    std::vector<f64> values = {
        1,  2,  // obj 0
        3,  4,  // obj 1
        5,  6,  // obj 2
        7,  8,  // obj 3
        9,  10, // obj 4
        11, 12, // obj 5
        13, 14, // obj 6
        15, 16, // obj 7
        17, 18, // obj 8
        19, 20, // obj 9
        21, 22, // obj 10
        23, 24, // obj 11
        25, 26, // obj 12
        27, 28, // obj 13
        29, 30, // obj 14
        31, 32, // obj 15
        33, 34, // obj 16
        35, 36, // obj 17
        37, 38, // obj 18
        39, 40, // obj 19
    };

    PatchDataField<f64> field = PatchDataField<f64>("test", 2, values.size() / 2);
    field.override(values, values.size());

    REQUIRE_EQUAL(field.get_buf().copy_to_stdvec(), values);

    std::vector<u32> to_be_removed = {0, 4, 8, 13, 12, 1};

    std::vector<f64> remaining_values = {
        5,  6,  // obj 2
        7,  8,  // obj 3
        11, 12, // obj 5
        13, 14, // obj 6
        15, 16, // obj 7
        19, 20, // obj 9
        21, 22, // obj 10
        23, 24, // obj 11
        29, 30, // obj 14
        31, 32, // obj 15
        33, 34, // obj 16
        35, 36, // obj 17
        37, 38, // obj 18
        39, 40, // obj 19
    };

    sham::DeviceBuffer<u32> to_be_removed_buf(
        to_be_removed.size(), shamsys::instance::get_compute_scheduler_ptr());
    to_be_removed_buf.copy_from_stdvec(to_be_removed);

    field.remove_ids(to_be_removed_buf, to_be_removed_buf.get_size());

    REQUIRE_EQUAL(field.get_buf().copy_to_stdvec(), remaining_values);
}

TestStart(Unittest, "shamrock/patch/PatchDataField::append_subset_to", testappendsubsetto, 1) {

    using T = f64;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    { // nvar = 1
        u32 len          = 10000;
        u32 nvar         = 1;
        std::string name = "testfield";

        PatchDataField<f64> field
            = PatchDataField<f64>::mock_field(0x1234, len, name, nvar, 0.0, 1000.0);

        std::vector<T> field_data = field.get_buf().copy_to_stdvec();

        std::vector<u32> idx_vec;
        for (u32 i = 0; i < len; i += 2) {
            idx_vec.push_back(i);
        }

        std::vector<T> ref;
        for (auto i : idx_vec) {
            ref.push_back(field_data[i]);
        }

        sham::DeviceBuffer<u32> idx_buf(idx_vec.size(), dev_sched);
        idx_buf.copy_from_stdvec(idx_vec);

        PatchDataField<f64> field_dest = PatchDataField<f64>(name, nvar, 0);
        field.append_subset_to(idx_buf, idx_buf.get_size(), field_dest);

        REQUIRE_EQUAL(field_dest.get_buf().copy_to_stdvec(), ref);
    }

    { // nvar = 2
        u32 len          = 10000;
        u32 nvar         = 2;
        std::string name = "testfield";

        PatchDataField<f64> field
            = PatchDataField<f64>::mock_field(0x1234, len, name, nvar, 0.0, 1000.0);

        std::vector<T> field_data = field.get_buf().copy_to_stdvec();

        std::vector<u32> idx_vec;
        for (u32 i = 0; i < len; i += 2) {
            idx_vec.push_back(i);
        }

        std::vector<T> ref;
        for (auto i : idx_vec) {
            ref.push_back(field_data[i * 2 + 0]);
            ref.push_back(field_data[i * 2 + 1]);
        }

        sham::DeviceBuffer<u32> idx_buf(idx_vec.size(), dev_sched);
        idx_buf.copy_from_stdvec(idx_vec);

        PatchDataField<f64> field_dest = PatchDataField<f64>(name, nvar, 0);
        field.append_subset_to(idx_buf, idx_buf.get_size(), field_dest);

        REQUIRE_EQUAL(field_dest.get_buf().copy_to_stdvec(), ref);
    }
}
