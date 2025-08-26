// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

TestStart(Unittest, "shamalgs/primitives/reduction/sum", test_reduction_sum, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test sum with simple integer values
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = std::accumulate(data.begin(), data.end(), 0);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with partial range
        std::vector<i32> data = {10, 20, 30, 40, 50, 60, 70};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 start    = 2;
        u32 end      = 5;
        i32 result   = shamalgs::primitives::sum(sched, buf, start, end);
        i32 expected = std::accumulate(data.begin() + start, data.begin() + end, 0);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with single element
        std::vector<i32> data = {42};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::sum(sched, buf, 0, 1);
        REQUIRE_EQUAL(result, 42);
    }

    {
        // Test sum with negative values
        std::vector<i32> data = {-10, -5, 0, 5, 10};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = std::accumulate(data.begin(), data.end(), 0);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with floating point values
        std::vector<f32> data = {1.5f, 2.7f, 3.14f, 4.2f, 5.9f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        f32 expected = std::accumulate(data.begin(), data.end(), 0.0f);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with double precision
        std::vector<f64> data = {1.123456789, 2.987654321, 3.141592653, 4.271828182};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        f64 expected = std::accumulate(data.begin(), data.end(), 0.0);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with unsigned integers
        std::vector<u32> data = {100, 200, 300, 400, 500};
        sham::DeviceBuffer<u32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        u32 expected = std::accumulate(data.begin(), data.end(), 0U);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with zeros
        std::vector<i32> data = {0, 0, 0, 0, 0};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 0);
    }

    {
        // Test sum with large values
        std::vector<u64> data = {1000000ULL, 2000000ULL, 3000000ULL, 4000000ULL};
        sham::DeviceBuffer<u64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u64 result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        u64 expected = std::accumulate(data.begin(), data.end(), 0ULL);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test sum with empty range (start == end)
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::sum(sched, buf, 2, 2);
        REQUIRE_EQUAL(result, 0); // Empty range should sum to 0
    }

    {
        // Test sum with random large dataset
        constexpr u32 size = 1000;
        std::vector<i32> data(size);
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_int_distribution<i32> dist(-1000, 1000);

        for (u32 i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }

        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::sum(sched, buf, 0, size);
        i32 expected = std::accumulate(data.begin(), data.end(), 0);
        REQUIRE_EQUAL(result, expected);
    }
}

TestStart(Unittest, "shamalgs/primitives/reduction/min", test_reduction_min, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test min with simple integer values
        std::vector<i32> data = {5, 2, 8, 1, 9, 3};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with partial range
        std::vector<i32> data = {10, 20, 30, 40, 50, 5, 60, 70};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 start    = 2;
        u32 end      = 6;
        i32 result   = shamalgs::primitives::min(sched, buf, start, end);
        i32 expected = *std::min_element(data.begin() + start, data.begin() + end);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with single element
        std::vector<i32> data = {42};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::min(sched, buf, 0, 1);
        REQUIRE_EQUAL(result, 42);
    }

    {
        // Test min with negative values
        std::vector<i32> data = {-10, -5, -20, -1, -15};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with floating point values
        std::vector<f32> data = {3.14f, 2.71f, 1.41f, 4.20f, 1.73f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        f32 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with double precision
        std::vector<f64> data = {1.123456789, 0.987654321, 2.141592653, 0.271828182};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        f64 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with unsigned integers
        std::vector<u32> data = {500, 200, 800, 100, 300};
        sham::DeviceBuffer<u32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        u32 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with all same values
        std::vector<i32> data = {7, 7, 7, 7, 7};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 7);
    }

    {
        // Test min at boundary positions
        std::vector<i32> data = {1, 10, 20, 30, 40, 2}; // Min at first and last positions
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 1);
    }

    {
        // Test min with large values
        std::vector<u64> data = {5000000ULL, 1000000ULL, 8000000ULL, 2000000ULL};
        sham::DeviceBuffer<u64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u64 result   = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        u64 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test min with random large dataset
        constexpr u32 size = 1000;
        std::vector<i32> data(size);
        std::mt19937 gen(54321); // Fixed seed for reproducibility
        std::uniform_int_distribution<i32> dist(-10000, 10000);

        for (u32 i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }

        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::min(sched, buf, 0, size);
        i32 expected = *std::min_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }
}

TestStart(Unittest, "shamalgs/primitives/reduction/max", test_reduction_max, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test max with simple integer values
        std::vector<i32> data = {5, 2, 8, 1, 9, 3};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with partial range
        std::vector<i32> data = {10, 20, 30, 80, 50, 5, 60, 70};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 start    = 2;
        u32 end      = 6;
        i32 result   = shamalgs::primitives::max(sched, buf, start, end);
        i32 expected = *std::max_element(data.begin() + start, data.begin() + end);
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with single element
        std::vector<i32> data = {42};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::max(sched, buf, 0, 1);
        REQUIRE_EQUAL(result, 42);
    }

    {
        // Test max with negative values
        std::vector<i32> data = {-10, -5, -20, -1, -15};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        i32 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with floating point values
        std::vector<f32> data = {3.14f, 2.71f, 1.41f, 4.20f, 1.73f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        f32 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with double precision
        std::vector<f64> data = {1.123456789, 0.987654321, 2.141592653, 0.271828182};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        f64 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with unsigned integers
        std::vector<u32> data = {500, 200, 800, 100, 300};
        sham::DeviceBuffer<u32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        u32 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with all same values
        std::vector<i32> data = {7, 7, 7, 7, 7};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 7);
    }

    {
        // Test max at boundary positions
        std::vector<i32> data = {90, 10, 20, 30, 40, 95}; // Max at first and last positions
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 95);
    }

    {
        // Test max with large values
        std::vector<u64> data = {5000000ULL, 1000000ULL, 8000000ULL, 2000000ULL};
        sham::DeviceBuffer<u64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u64 result   = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        u64 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test max with random large dataset
        constexpr u32 size = 1000;
        std::vector<i32> data(size);
        std::mt19937 gen(98765); // Fixed seed for reproducibility
        std::uniform_int_distribution<i32> dist(-10000, 10000);

        for (u32 i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }

        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result   = shamalgs::primitives::max(sched, buf, 0, size);
        i32 expected = *std::max_element(data.begin(), data.end());
        REQUIRE_EQUAL(result, expected);
    }
}

TestStart(Unittest, "shamalgs/primitives/reduction/edge_cases", test_reduction_edge_cases, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test with extreme values for integers
        std::vector<i32> data
            = {std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max(), 0, -1, 1};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 sum_result = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        // Note: This may overflow, which is expected behavior for integer arithmetic
        i32 expected_sum = std::accumulate(data.begin(), data.end(), static_cast<i32>(0));
        REQUIRE_EQUAL(sum_result, expected_sum);

        i32 min_result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(min_result, std::numeric_limits<i32>::min());

        i32 max_result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(max_result, std::numeric_limits<i32>::max());
    }

    {
        // Test with extreme values for floating point
        std::vector<f32> data = {
            std::numeric_limits<f32>::lowest(), std::numeric_limits<f32>::max(), 0.0f, -1.0f, 1.0f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 sum_result   = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        f32 expected_sum = std::accumulate(data.begin(), data.end(), 0.0f);
        REQUIRE_EQUAL(sum_result, expected_sum);

        f32 min_result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(min_result, std::numeric_limits<f32>::lowest());

        f32 max_result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(max_result, std::numeric_limits<f32>::max());
    }

    {
        // Test with powers of 2 sizes (edge case for reduction algorithms)
        for (u32 array_size : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U}) {
            std::vector<i32> data(array_size);

            // Fill with sequential values
            for (u32 i = 0; i < array_size; ++i) {
                data[i] = static_cast<i32>(i + 1);
            }

            sham::DeviceBuffer<i32> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            i32 sum_result   = shamalgs::primitives::sum(sched, buf, 0, array_size);
            i32 expected_sum = static_cast<i32>(array_size * (array_size + 1) / 2);
            REQUIRE_EQUAL(sum_result, expected_sum);

            i32 min_result = shamalgs::primitives::min(sched, buf, 0, array_size);
            REQUIRE_EQUAL(min_result, 1);

            i32 max_result = shamalgs::primitives::max(sched, buf, 0, array_size);
            REQUIRE_EQUAL(max_result, static_cast<i32>(array_size));
        }
    }

    {
        // Test with various partial ranges
        std::vector<i32> data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        // Test different range sizes
        for (u32 start = 0; start <= 5; ++start) {
            for (u32 end = start; end <= data.size(); ++end) {
                if (start == end) {
                    // Empty range - only test sum (min/max are undefined for empty ranges)
                    i32 sum_result = shamalgs::primitives::sum(sched, buf, start, end);
                    REQUIRE_EQUAL(sum_result, 0);

                    // min should throw exception for empty ranges
                    REQUIRE_EXCEPTION_THROW(
                        shamalgs::primitives::min(sched, buf, start, end), std::invalid_argument);

                    // max should throw exception for empty ranges
                    REQUIRE_EXCEPTION_THROW(
                        shamalgs::primitives::max(sched, buf, start, end), std::invalid_argument);
                } else {
                    // Non-empty range
                    i32 sum_result   = shamalgs::primitives::sum(sched, buf, start, end);
                    i32 expected_sum = std::accumulate(data.begin() + start, data.begin() + end, 0);
                    REQUIRE_EQUAL(sum_result, expected_sum);

                    i32 min_result   = shamalgs::primitives::min(sched, buf, start, end);
                    i32 expected_min = *std::min_element(data.begin() + start, data.begin() + end);
                    REQUIRE_EQUAL(min_result, expected_min);

                    i32 max_result   = shamalgs::primitives::max(sched, buf, start, end);
                    i32 expected_max = *std::max_element(data.begin() + start, data.begin() + end);
                    REQUIRE_EQUAL(max_result, expected_max);
                }
            }
        }
    }

    {
        // Test with very small floating point differences
        std::vector<f64> data = {1.0000000001, 1.0000000002, 1.0000000003, 1.0000000004};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 min_result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(min_result, 1.0000000001);

        f64 max_result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(max_result, 1.0000000004);
    }

    {
        // Test with different data types: u64
        std::vector<u64> data = {18446744073709551615ULL, 0ULL, 9223372036854775808ULL};
        sham::DeviceBuffer<u64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u64 min_result = shamalgs::primitives::min(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(min_result, 0ULL);

        u64 max_result = shamalgs::primitives::max(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(max_result, 18446744073709551615ULL);
    }

    {
        // Test empty range exception for min function
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        REQUIRE_EXCEPTION_THROW(shamalgs::primitives::min(sched, buf, 2, 2), std::invalid_argument);
    }

    {
        // Test empty range exception for max function
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        REQUIRE_EXCEPTION_THROW(shamalgs::primitives::max(sched, buf, 3, 3), std::invalid_argument);
    }

    {
        // Test invalid range where start_id > end_id for all functions
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        // sum should return 0 for invalid ranges (start > end)
        i32 sum_result = shamalgs::primitives::sum(sched, buf, 4, 2);
        REQUIRE_EQUAL(sum_result, 0);

        // min should throw exception for invalid ranges
        REQUIRE_EXCEPTION_THROW(shamalgs::primitives::min(sched, buf, 4, 2), std::invalid_argument);

        // max should throw exception for invalid ranges
        REQUIRE_EXCEPTION_THROW(shamalgs::primitives::max(sched, buf, 4, 2), std::invalid_argument);
    }
}
