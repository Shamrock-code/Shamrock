// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/lower_bound.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <random>
#include <vector>

TestStart(
    Unittest, "shamalgs/primitives/binary_search_lower_bound", test_binary_search_lower_bound, 1) {

    {
        // Test with empty array
        std::vector<i32> data = {};
        u32 result      = shamalgs::primitives::binary_search_lower_bound(data.data(), 0, 0, 5);
        auto std_result = std::lower_bound(data.begin(), data.end(), 5) - data.begin();
        REQUIRE_EQUAL(result, static_cast<u32>(std_result));
    }

    {
        // Test with single element - value found
        std::vector<i32> data = {10};
        u32 result      = shamalgs::primitives::binary_search_lower_bound(data.data(), 0, 1, 10);
        auto std_result = std::lower_bound(data.begin(), data.end(), 10) - data.begin();
        REQUIRE_EQUAL(result, static_cast<u32>(std_result));
    }

    {
        // Test with single element - value less than element
        std::vector<i32> data = {10};
        u32 result      = shamalgs::primitives::binary_search_lower_bound(data.data(), 0, 1, 5);
        auto std_result = std::lower_bound(data.begin(), data.end(), 5) - data.begin();
        REQUIRE_EQUAL(result, static_cast<u32>(std_result));
    }

    {
        // Test with single element - value greater than element
        std::vector<i32> data = {10};
        u32 result      = shamalgs::primitives::binary_search_lower_bound(data.data(), 0, 1, 15);
        auto std_result = std::lower_bound(data.begin(), data.end(), 15) - data.begin();
        REQUIRE_EQUAL(result, static_cast<u32>(std_result));
    }

    {
        // Test with sorted array - values found
        std::vector<i32> data = {1, 3, 5, 7, 9, 11, 13, 15};

        for (auto value : data) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with sorted array - values not found
        std::vector<i32> data        = {1, 3, 5, 7, 9, 11, 13, 15};
        std::vector<i32> test_values = {0, 2, 4, 6, 8, 10, 12, 14, 16, 20};

        for (auto value : test_values) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with duplicates
        std::vector<i32> data = {1, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6};

        for (i32 value = 0; value <= 7; ++value) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with all same elements
        std::vector<i32> data = {5, 5, 5, 5, 5, 5, 5, 5};

        std::vector<i32> test_values = {3, 5, 7};
        for (auto value : test_values) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with partial range
        std::vector<i32> data = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};
        u32 first             = 2;
        u32 last              = 7;

        for (i32 value = 4; value <= 14; ++value) {
            u32 result
                = shamalgs::primitives::binary_search_lower_bound(data.data(), first, last, value);
            auto std_result
                = std::lower_bound(data.begin() + first, data.begin() + last, value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with unsigned integers
        std::vector<u32> data = {0, 5, 10, 15, 20, 25, 30};

        for (u32 value = 0; value <= 35; value += 3) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with floating point values
        std::vector<f32> data        = {-2.5f, -1.0f, 0.0f, 1.5f, 3.14f, 5.7f, 10.0f};
        std::vector<f32> test_values = {-3.0f, -1.0f, 0.5f, 3.14f, 7.0f, 15.0f};

        for (auto value : test_values) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
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

        // Test with random values
        std::uniform_int_distribution<i32> test_dist(0, 11000);
        for (u32 i = 0; i < 100; ++i) {
            i32 value  = test_dist(gen);
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with powers of 2 sizes (edge case for binary algorithms)
        for (u32 array_size : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U}) {
            std::vector<i32> data(array_size);

            // Fill with values 0, 2, 4, 6, ...
            for (u32 i = 0; i < array_size; ++i) {
                data[i] = static_cast<i32>(i * 2);
            }

            // Test with various values
            for (i32 value = -1; value <= static_cast<i32>(array_size * 2 + 1); ++value) {
                u32 result = shamalgs::primitives::binary_search_lower_bound(
                    data.data(), 0, static_cast<u32>(data.size()), value);
                auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
                REQUIRE_EQUAL(result, static_cast<u32>(std_result));
            }
        }
    }

    {
        // Test with 64-bit integers
        std::vector<u64> data = {100ULL, 1000ULL, 10000ULL, 100000ULL, 1000000ULL};
        std::vector<u64> test_values
            = {50ULL, 100ULL, 500ULL, 1000ULL, 50000ULL, 100000ULL, 2000000ULL};

        for (auto value : test_values) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test boundary conditions
        std::vector<i32> data = {10, 20, 30, 40, 50};

        // Test with minimum and maximum values
        for (i32 value : {INT32_MIN, 5, 10, 55, INT32_MAX}) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }

    {
        // Test with negative numbers
        std::vector<i32> data = {-100, -50, -10, -5, 0, 5, 10, 50, 100};

        for (i32 value = -120; value <= 120; value += 15) {
            u32 result = shamalgs::primitives::binary_search_lower_bound(
                data.data(), 0, static_cast<u32>(data.size()), value);
            auto std_result = std::lower_bound(data.begin(), data.end(), value) - data.begin();
            REQUIRE_EQUAL(result, static_cast<u32>(std_result));
        }
    }
}
