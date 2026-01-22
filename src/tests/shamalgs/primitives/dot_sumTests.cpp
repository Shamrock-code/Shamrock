// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/dot_sum.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <limits>
#include <random>
#include <vector>

TestStart(Unittest, "shamalgs/primitives/dot_sum/scalar_types", test_dot_sum_scalar_types, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test dot_sum with simple integer values (scalars)
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        // For scalars, dot(x, x) = x * x
        i32 expected = 0;
        for (auto val : data) {
            expected += val * val;
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with partial range
        std::vector<i32> data = {10, 20, 30, 40, 50, 60, 70};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 start  = 2;
        u32 end    = 5;
        i32 result = shamalgs::primitives::dot_sum(buf, start, end);

        i32 expected = 0;
        for (u32 i = start; i < end; ++i) {
            expected += data[i] * data[i];
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with single element
        std::vector<i32> data = {42};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 0, 1);
        REQUIRE_EQUAL(result, 42 * 42);
    }

    {
        // Test dot_sum with negative values
        std::vector<i32> data = {-3, -2, 0, 2, 3};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        i32 expected = 0;
        for (auto val : data) {
            expected += val * val;
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with floating point values
        std::vector<f32> data = {1.5f, 2.7f, 3.14f, 4.2f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        for (u32 i = 0; i < data.size(); ++i) {
            data[i] = data[i] * data[i];
        }
        buf.copy_from_stdvec(data);

        f32 expected = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with double precision
        std::vector<f64> data = {1.123456789, 2.987654321, 3.141592653};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        for (u32 i = 0; i < data.size(); ++i) {
            data[i] = data[i] * data[i];
        }
        buf.copy_from_stdvec(data);

        f64 expected = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with unsigned integers
        std::vector<u32> data = {100, 200, 300, 400};
        sham::DeviceBuffer<u32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        u32 expected = 0;
        for (auto val : data) {
            expected += val * val;
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with zeros
        std::vector<i32> data = {0, 0, 0, 0, 0};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, 0);
    }
}

TestStart(Unittest, "shamalgs/primitives/dot_sum/vector_types", test_dot_sum_vector_types, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test dot_sum with 2D vectors
        std::vector<f32_2> data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        sham::DeviceBuffer<f32_2> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        // For 2D vectors, dot(v, v) = v.x*v.x + v.y*v.y
        f32 expected = 0.0f;
        for (const auto &vec : data) {
            expected += vec.x() * vec.x() + vec.y() * vec.y();
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with 3D vectors
        std::vector<f64_3> data = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        sham::DeviceBuffer<f64_3> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        // For 3D vectors, dot(v, v) = v.x*v.x + v.y*v.y + v.z*v.z
        f64 expected = 0.0;
        for (const auto &vec : data) {
            expected += vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z();
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with 4D vectors
        std::vector<f32_4> data = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
        sham::DeviceBuffer<f32_4> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        // For 4D vectors, dot(v, v) = v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w
        f32 expected = 0.0f;
        for (const auto &vec : data) {
            expected
                += vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z() + vec.w() * vec.w();
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with integer 3D vectors
        std::vector<i32_3> data = {{1, 2, 3}, {-1, -2, -3}, {0, 0, 0}, {5, 0, -5}};
        sham::DeviceBuffer<i32_3> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        i32 expected = 0;
        for (const auto &vec : data) {
            expected += vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z();
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with partial range on vectors
        std::vector<f64_2> data = {{1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}};
        sham::DeviceBuffer<f64_2> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        u32 start  = 1;
        u32 end    = 4;
        f64 result = shamalgs::primitives::dot_sum(buf, start, end);

        f64 expected = 0.0;
        for (u32 i = start; i < end; ++i) {
            const auto &vec = data[i];
            expected += vec.x() * vec.x() + vec.y() * vec.y();
        }
        REQUIRE_EQUAL(result, expected);
    }
}

TestStart(Unittest, "shamalgs/primitives/dot_sum/edge_cases", test_dot_sum_edge_cases, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test dot_sum with empty range (start == end)
        std::vector<i32> data = {1, 2, 3, 4, 5};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        i32 result = shamalgs::primitives::dot_sum(buf, 2, 2);
        REQUIRE_EQUAL(result, 0); // Empty range should return 0
    }

    {
        // Test dot_sum with invalid range (start > end)
        std::vector<f32> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        // This should throw an exception
        REQUIRE_EXCEPTION_THROW(shamalgs::primitives::dot_sum(buf, 4, 2), std::invalid_argument);
    }

    {
        // Test dot_sum with special values for integers
        std::vector<i32> data = {0, -1, 1};
        sham::DeviceBuffer<i32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        // Note: This may overflow, which is expected behavior for integer arithmetic
        i32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        i64 expected = 0; // Use larger type to avoid overflow in calculation
        for (auto val : data) {
            expected += static_cast<i64>(val) * static_cast<i64>(val);
        }
        // Cast back to i32 to match overflow behavior
        REQUIRE_EQUAL(result, static_cast<i32>(expected));
    }

    {
        // Test dot_sum with extreme values for floating point
        std::vector<f32> data = {
            std::numeric_limits<f32>::max(), std::numeric_limits<f32>::lowest(), 0.0f, -1.0f, 1.0f};
        sham::DeviceBuffer<f32> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        // This will likely result in infinity due to max^2
        f32 expected = 0.0f;
        for (auto val : data) {
            expected += val * val;
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with powers of 2 sizes (edge case for reduction algorithms)
        for (u32 array_size : {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U, 256U, 512U, 1024U}) {
            std::vector<f64> data(array_size);

            // Fill with sequential values
            for (u32 i = 0; i < array_size; ++i) {
                data[i] = static_cast<f64>(i + 1);
            }

            sham::DeviceBuffer<f64> buf(data.size(), sched);
            buf.copy_from_stdvec(data);

            f64 result = shamalgs::primitives::dot_sum(buf, 0, array_size);

            // Expected: sum of squares from 1 to array_size
            f64 expected = 0.0;
            for (u32 i = 1; i <= array_size; ++i) {
                expected += static_cast<f64>(i) * static_cast<f64>(i);
            }
            REQUIRE_EQUAL(result, expected);
        }
    }

    {
        // Test dot_sum with very small floating point values
        std::vector<f64> data = {1e-10, 2e-10, 3e-10, 4e-10};
        sham::DeviceBuffer<f64> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        for (u32 i = 0; i < data.size(); ++i) {
            data[i] = data[i] * data[i];
        }
        buf.copy_from_stdvec(data);

        f64 expected = shamalgs::primitives::sum(sched, buf, 0, static_cast<u32>(data.size()));
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with mixed positive and negative vectors
        std::vector<f32_3> data = {{1.0f, -2.0f, 3.0f}, {-4.0f, 5.0f, -6.0f}, {0.0f, 0.0f, 0.0f}};
        sham::DeviceBuffer<f32_3> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f32 result = shamalgs::primitives::dot_sum(buf, 0, static_cast<u32>(data.size()));

        f32 expected = 0.0f;
        for (const auto &vec : data) {
            expected += vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z();
        }
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test dot_sum with large dataset and random values
        constexpr u32 size = 1000;
        std::vector<f64_2> data(size);
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<f64> dist(-100.0, 100.0);

        for (u32 i = 0; i < size; ++i) {
            data[i] = {dist(gen), dist(gen)};
        }

        sham::DeviceBuffer<f64_2> buf(data.size(), sched);
        buf.copy_from_stdvec(data);

        f64 result = shamalgs::primitives::dot_sum(buf, 0, size);

        f64 expected = 0.0;
        for (const auto &vec : data) {
            expected += vec.x() * vec.x() + vec.y() * vec.y();
        }
        REQUIRE_EQUAL(result, expected);
    }
}
