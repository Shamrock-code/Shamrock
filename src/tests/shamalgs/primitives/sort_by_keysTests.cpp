// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/mock_vector.hpp"
#include "shamalgs/primitives/sort_by_keys.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <utility>
#include <vector>

NEW_TEST(Unittest, "shamalgs/primitives/sort_by_keys", 1) {

    auto test_run = []() {
        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        { // empty buffers
            sham::DeviceBuffer<u32> keys(0, sched);
            sham::DeviceBuffer<u32> values(0, sched);

            shamalgs::primitives::sort_by_keys(keys, values, 0);

            REQUIRE_EQUAL(keys.copy_to_stdvec(), std::vector<u32>{});
            REQUIRE_EQUAL(values.copy_to_stdvec(), std::vector<u32>{});
        }

        { // single element
            sham::DeviceBuffer<u32> keys(1, sched);
            keys.copy_from_stdvec({42});
            sham::DeviceBuffer<u32> values(1, sched);
            values.copy_from_stdvec({1});

            shamalgs::primitives::sort_by_keys(keys, values, 1);

            REQUIRE_EQUAL(keys.copy_to_stdvec(), std::vector<u32>{42});
            REQUIRE_EQUAL(values.copy_to_stdvec(), std::vector<u32>{1});
        }

        { // odd, non power-of-2 length, already sorted
            std::vector<u32> key_data   = {1, 2, 3, 4, 5};
            std::vector<u32> value_data = {10, 20, 30, 40, 50};

            sham::DeviceBuffer<u32> keys(key_data.size(), sched);
            keys.copy_from_stdvec(key_data);
            sham::DeviceBuffer<u32> values(value_data.size(), sched);
            values.copy_from_stdvec(value_data);

            shamalgs::primitives::sort_by_keys(keys, values, key_data.size());

            REQUIRE_EQUAL(keys.copy_to_stdvec(), key_data);
            REQUIRE_EQUAL(values.copy_to_stdvec(), value_data);
        }

        { // odd, non power-of-2 length, reverse sorted
            std::vector<u32> key_data   = {5, 4, 3, 2, 1};
            std::vector<u32> value_data = {50, 40, 30, 20, 10};

            sham::DeviceBuffer<u32> keys(key_data.size(), sched);
            keys.copy_from_stdvec(key_data);
            sham::DeviceBuffer<u32> values(value_data.size(), sched);
            values.copy_from_stdvec(value_data);

            shamalgs::primitives::sort_by_keys(keys, values, key_data.size());

            std::vector<u32> expected_key   = {1, 2, 3, 4, 5};
            std::vector<u32> expected_value = {10, 20, 30, 40, 50};
            REQUIRE_EQUAL(keys.copy_to_stdvec(), expected_key);
            REQUIRE_EQUAL(values.copy_to_stdvec(), expected_value);
        }

        { // random order, prime (non power-of-2) length
            u32 len = 4099;
            std::vector<u32> key_data
                = shamalgs::primitives::mock_vector<u32>(0x123, len, 0, 1000000);
            std::vector<u32> value_data(len);
            for (u32 i = 0; i < len; ++i) {
                value_data[i] = i;
            }

            std::vector<std::pair<u32, u32>> expected_zip(len);
            for (u32 i = 0; i < len; ++i) {
                expected_zip[i] = {key_data[i], value_data[i]};
            }
            std::sort(expected_zip.begin(), expected_zip.end());

            sham::DeviceBuffer<u32> keys(len, sched);
            keys.copy_from_stdvec(key_data);
            sham::DeviceBuffer<u32> values(len, sched);
            values.copy_from_stdvec(value_data);

            shamalgs::primitives::sort_by_keys(keys, values, len);

            std::vector<u32> result_key   = keys.copy_to_stdvec();
            std::vector<u32> result_value = values.copy_to_stdvec();

            // the mapping between a key and its value must be preserved, and the keys must
            // end up sorted (ties may be reordered, since the implementation is not required
            // to be stable)
            REQUIRE(std::is_sorted(result_key.begin(), result_key.end()));

            std::vector<std::pair<u32, u32>> result_zip(len);
            for (u32 i = 0; i < len; ++i) {
                result_zip[i] = {result_key[i], result_value[i]};
            }
            std::sort(result_zip.begin(), result_zip.end());

            // compare the (key, value) multisets without relying on a fmt formatter for
            // std::pair : flatten both sorted zips back into plain u32 vectors
            std::vector<u32> expected_key_sorted(len), expected_val_sorted(len);
            std::vector<u32> result_key_sorted(len), result_val_sorted(len);
            for (u32 i = 0; i < len; ++i) {
                expected_key_sorted[i] = expected_zip[i].first;
                expected_val_sorted[i] = expected_zip[i].second;
                result_key_sorted[i]   = result_zip[i].first;
                result_val_sorted[i]   = result_zip[i].second;
            }
            REQUIRE_EQUAL(result_key_sorted, expected_key_sorted);
            REQUIRE_EQUAL(result_val_sorted, expected_val_sorted);
        }
    };

    if (!shamalgs::primitives::impl::is_impl_set_sort_by_keys()) {
        shamalgs::primitives::impl::autoselect_impl_sort_by_keys();
    }
    auto current_impl = shamalgs::primitives::impl::get_current_impl_sort_by_keys();

    for (const std::string &impl :
         shamalgs::primitives::impl::get_default_impl_list_sort_by_keys()) {
        shamalgs::primitives::impl::set_impl_sort_by_keys(impl);
        shamlog_info_ln("tests", "testing implementation:", impl);
        test_run();
    }

    // reset to default
    shamalgs::primitives::impl::set_impl_sort_by_keys(current_impl);
}
