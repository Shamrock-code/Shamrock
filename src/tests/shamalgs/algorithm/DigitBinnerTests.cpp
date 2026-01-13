// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamalgs/details/algorithm/DigitBinner.hpp"
#include "shamalgs/random.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

template<u32 group_size, u32 digit_len, class T>
inline void add_dset(std::string sname) {
    auto res = shambase::benchmark_pow_len(
        [](u32 nobj) {
            sycl::queue &q = shamsys::instance::get_compute_queue();

            sycl::buffer<u32> buf_key
                = shamalgs::random::mock_buffer<u32>(0x111, nobj, 0, 1U << 31U);

            u32 group_cnt = shambase::group_count(nobj, group_size);

            u32 corrected_len = group_cnt * group_size;

            using namespace shamalgs::algorithm::details;

            using Binner = DigitBinner<T, digit_len>;

            shamsys::instance::get_compute_queue().wait();

            return shambase::timeit(
                [&]() {
                    sycl::buffer<u32> digit_histogram
                        = Binner::template make_digit_histogram<group_size>(q, buf_key, nobj);
                    shamsys::instance::get_compute_queue().wait();
                },
                10);
        },
        10,
        1e8,
        1.1);

    auto &dat_test = shamtest::test_data().new_dataset(sname);

    dat_test.add_data("Nobj", res.counts);
    dat_test.add_data("time", res.times);
}

TestStart(
    Benchmark, "shamalgs/algorithm/details/DigitBinner:benchmark", benchmark_digit_binner, 1) {

    // add_dset<128, 1, u32>("group size = 128, bits = 1");
    // add_dset<256, 1, u32>("group size = 256, bits = 1");
    add_dset<512, 1, u32>("group size = 512, bits = 1");

    // add_dset<128, 2, u32>("group size = 128, bits = 2");
    // add_dset<256, 2, u32>("group size = 256, bits = 2");
    add_dset<512, 2, u32>("group size = 512, bits = 2");

    // add_dset<128, 4, u32>("group size = 128, bits = 4");
    // add_dset<256, 4, u32>("group size = 256, bits = 4");
    add_dset<512, 4, u32>("group size = 512, bits = 4");

    // add_dset<128, 8, u32>("group size = 128, bits = 8");
    // add_dset<256, 8, u32>("group size = 256, bits = 8");
    add_dset<512, 8, u32>("group size = 512, bits = 8");
}
