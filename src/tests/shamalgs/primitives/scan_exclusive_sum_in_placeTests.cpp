// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(
    Unittest,
    "shamalgs/primitives/scan_exclusive_sum_in_place",
    test_scan_exclusive_sum_in_place,
    1) {

    auto test_run = []() {
        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        { // empty dataset
            sham::DeviceBuffer<u32> buf(0, sched);

            shamalgs::primitives::scan_exclusive_sum_in_place(buf, 0);

            REQUIRE_EQUAL(buf.copy_to_stdvec(), std::vector<u32>{});
        }

        { // Larger scan than buffer
            sham::DeviceBuffer<u32> buf(2, sched);
            REQUIRE_EXCEPTION_THROW(
                shamalgs::primitives::scan_exclusive_sum_in_place(buf, 10), std::invalid_argument);
        }

        { // small dataset
            std::vector<u32> data = {1, 2, 3, 4, 5};
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            auto ref = shamalgs::numeric::scan_exclusive(sched, buf, data.size());
            shamalgs::primitives::scan_exclusive_sum_in_place(buf, data.size());

            REQUIRE_EQUAL(buf.copy_to_stdvec(), ref.copy_to_stdvec());
        }

        { // large dataset
            std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(0x111, 10000000, 0, 10);
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            auto ref = shamalgs::numeric::scan_exclusive(sched, buf, data.size());
            shamalgs::primitives::scan_exclusive_sum_in_place(buf, data.size());

            REQUIRE_EQUAL(buf.copy_to_stdvec(), ref.copy_to_stdvec());
        }

        { // partial scan
            u32 len               = 10'000'000;
            u32 len_scan          = len / 2;
            std::vector<u32> data = shamalgs::primitives::mock_vector<u32>(0x111, len, 0, 10);
            sham::DeviceBuffer<u32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            auto ref = shamalgs::numeric::scan_exclusive(sched, buf, len_scan);
            shamalgs::primitives::scan_exclusive_sum_in_place(buf, len_scan);

            std::vector<u32> expected = ref.copy_to_stdvec();

            for (auto it = len_scan; it < len; ++it) {
                expected.push_back(data[it]);
            }

            REQUIRE_EQUAL(buf.copy_to_stdvec(), expected);
        }
    };

    for (std::string impl :
         shamalgs::primitives::impl::get_impl_list_scan_exclusive_sum_in_place()) {
        shamalgs::primitives::impl::set_impl_scan_exclusive_sum_in_place(impl);
        test_run();
    }

    // reset to default
    shamalgs::primitives::impl::set_impl_scan_exclusive_sum_in_place_default();
}
