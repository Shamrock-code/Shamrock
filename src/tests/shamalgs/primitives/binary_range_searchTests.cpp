// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/string.hpp"
#include "shamalgs/primitives/binary_range_search.hpp"
#include "shamcomm/logs.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <random>
#include <vector>

template<class Tkey>
void test_binary_range_search(const std::vector<Tkey> &data, Tkey value_min, Tkey value_max) {

    if (!(value_min <= value_max)) {
        return;
    }
    if (!((data.size() > 0) ? data[0] <= value_min : true)) {
        return;
    }
    if (!((data.size() > 0) ? data[data.size() - 1] >= value_max : true)) {
        return;
    }

    u32 inf, sup;
    shamalgs::primitives::binary_range_search(
        data.data(), 0, data.size(), value_min, value_max, inf, sup);

    u32 expected_inf = 0, expected_sup = data.size();

    for (u32 i = 0; i < data.size(); ++i) {
        if (data[i] <= value_min) {
            expected_inf = i;
            if (data[i] == value_min) {
                break;
            }
        }
    }

    for (i32 i = data.size() - 1; i >= 0; --i) {
        if (data[i] >= value_max) {
            expected_sup = i;
            if (data[i] == value_max) {
                break;
            }
        }
    }

    // shamcomm::logs::raw_ln(shambase::format(
    //    "--------\ndata = {}\nvalue_min = {}\nvalue_max = {}\ninf = {}\nsup = {}\nexpected_inf = "
    //    "{}\nexpected_sup = {}\nvalid_inf = {}\nvalid_sup = {}",
    //    data,
    //    value_min,
    //    value_max,
    //    inf,
    //    sup,
    //    expected_inf,
    //    expected_sup,
    //    inf == expected_inf,
    //    sup == expected_sup));

    REQUIRE_EQUAL(expected_inf, inf);
    REQUIRE_EQUAL(expected_sup, sup);
}

TestStart(Unittest, "shamalgs/primitives/binary_range_search", test_binary_range_search, 1) {

    {
        // Test with empty array
        std::vector<i32> data = {};
        i32 value_min         = 5;
        i32 value_max         = 10;

        test_binary_range_search(data, 5, 10);
    }

    {
        // Test with single element - value range includes element
        std::vector<i32> data = {10};
        i32 value_min         = 5;
        i32 value_max         = 15;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with single element - value range excludes element (too low)
        std::vector<i32> data = {10};
        i32 value_min         = 1;
        i32 value_max         = 5;

        // Range [1, 5] is before element 10
        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with single element - value range excludes element (too high)
        std::vector<i32> data = {10};
        i32 value_min         = 15;
        i32 value_max         = 20;

        // Range [15, 20] is after element 10
        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with sorted array - range includes multiple elements
        std::vector<i32> data = {1, 3, 5, 7, 9, 11, 13, 15};
        i32 value_min         = 5;
        i32 value_max         = 11;

        // Range [5, 11] should include elements 5, 7, 9, 11 (indices 2, 3, 4, 5)
        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with sorted array - range includes no elements (gap)
        std::vector<i32> data = {1, 3, 5, 7, 9, 11, 13, 15};
        i32 value_min         = 8;
        i32 value_max         = 8;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with duplicates
        std::vector<i32> data = {1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6};
        i32 value_min         = 2;
        i32 value_max         = 4;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with exact element range
        std::vector<i32> data = {1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6};
        i32 value_min         = 5;
        i32 value_max         = 5;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with range before all elements
        std::vector<i32> data = {10, 20, 30, 40, 50};
        i32 value_min         = 1;
        i32 value_max         = 5;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with range after all elements
        std::vector<i32> data = {10, 20, 30, 40, 50};
        i32 value_min         = 60;
        i32 value_max         = 70;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with range covering all elements
        std::vector<i32> data = {10, 20, 30, 40, 50};
        i32 value_min         = 5;
        i32 value_max         = 55;

        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test with large random dataset
        constexpr u32 size = 1000;
        std::vector<i32> data(size);

        // Generate sorted random data
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<i32> dist(1, 10000);

        for (u32 i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        std::sort(data.begin(), data.end());

        // Test with random ranges
        std::uniform_int_distribution<i32> range_dist(1, 9000);
        for (u32 test = 0; test < 50; ++test) {
            i32 min_val = range_dist(gen);
            i32 max_val = min_val + range_dist(gen) % 1000; // Ensure max_val >= min_val

            test_binary_range_search(data, min_val, max_val);
        }
    }

    {
        // Test with powers of 2 sizes
        for (u32 array_size : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}) {
            std::vector<i32> data(array_size);

            // Fill with values 0, 2, 4, 6, ...
            for (u32 i = 0; i < array_size; ++i) {
                data[i] = static_cast<i32>(i * 2);
            }

            // Test various ranges
            for (i32 min_val = -1; min_val <= static_cast<i32>(array_size * 2); min_val += 3) {
                for (i32 max_val = min_val; max_val <= static_cast<i32>(array_size * 2 + 1);
                     max_val += 4) {
                    test_binary_range_search(data, min_val, max_val);
                }
            }
        }
    }

    {
        // Test with negative numbers
        std::vector<i32> data = {-100, -50, -10, -5, 0, 5, 10, 50, 100};
        i32 value_min         = -20;
        i32 value_max         = 20;

        // Range [-20, 20] should include -10, -5, 0, 5, 10
        test_binary_range_search(data, value_min, value_max);
    }

    {
        // Test boundary values
        std::vector<i32> data = {10, 20, 30, 40, 50};
        i32 value_min         = 20;
        i32 value_max         = 40;

        // Test with exact boundary match

        test_binary_range_search(data, value_min, value_max);
    }
}
