// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/equals.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/primitives/equals", test_equals_primitive, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test with identical integer buffers
        constexpr u32 size    = 10;
        std::vector<i32> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(result);
    }

    {
        // Test with identical float buffers
        constexpr u32 size    = 5;
        std::vector<f32> data = {1.0f, 2.5f, 3.14f, 4.7f, 5.9f};

        sham::DeviceBuffer<f32> buf1(size, sched);
        sham::DeviceBuffer<f32> buf2(size, sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(result);
    }

    {
        // Test with single element
        constexpr u32 size    = 1;
        std::vector<i32> data = {42};

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(result);
    }

    {
        // Test with different integer buffers
        constexpr u32 size     = 5;
        std::vector<i32> data1 = {1, 2, 3, 4, 5};
        std::vector<i32> data2 = {1, 2, 3, 4, 6}; // Last element different

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(!result);
    }

    {
        // Test with completely different buffers
        constexpr u32 size     = 4;
        std::vector<i32> data1 = {10, 20, 30, 40};
        std::vector<i32> data2 = {50, 60, 70, 80};

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(!result);
    }

    {
        // Test with first element different
        constexpr u32 size     = 3;
        std::vector<i32> data1 = {1, 2, 3};
        std::vector<i32> data2 = {0, 2, 3}; // First element different

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(!result);
    }

    {
        // Test partial comparison where first part is identical
        constexpr u32 total_size   = 10;
        constexpr u32 compare_size = 5;
        std::vector<i32> data1     = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<i32> data2     = {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}; // First 5 identical

        sham::DeviceBuffer<i32> buf1(total_size, sched);
        sham::DeviceBuffer<i32> buf2(total_size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, compare_size);
        REQUIRE(result);
    }

    {
        // Test partial comparison where first part is different
        constexpr u32 total_size   = 8;
        constexpr u32 compare_size = 3;
        std::vector<i32> data1     = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<i32> data2     = {1, 2, 4, 4, 5, 6, 7, 8}; // Third element different

        sham::DeviceBuffer<i32> buf1(total_size, sched);
        sham::DeviceBuffer<i32> buf2(total_size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, compare_size);
        REQUIRE(!result);
    }

    {
        // Test with same buffer reference
        constexpr u32 size    = 7;
        std::vector<i32> data = {10, 20, 30, 40, 50, 60, 70};

        sham::DeviceBuffer<i32> buf(size, sched);
        buf.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf, buf, size);
        REQUIRE(result);
    }

    {
        // Test with same buffer reference and partial comparison
        constexpr u32 total_size   = 6;
        constexpr u32 compare_size = 3;
        std::vector<i32> data      = {100, 200, 300, 400, 500, 600};

        sham::DeviceBuffer<i32> buf(total_size, sched);
        buf.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf, buf, compare_size);
        REQUIRE(result);
    }

    {
        // Test with buffers of different sizes (should return false)
        std::vector<i32> data1 = {1, 2, 3, 4, 5};
        std::vector<i32> data2 = {1, 2, 3};

        sham::DeviceBuffer<i32> buf1(data1.size(), sched);
        sham::DeviceBuffer<i32> buf2(data2.size(), sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2);
        REQUIRE(!result);
    }

    {
        // Test with buffers of same size and identical content
        std::vector<i32> data = {7, 8, 9, 10};

        sham::DeviceBuffer<i32> buf1(data.size(), sched);
        sham::DeviceBuffer<i32> buf2(data.size(), sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2);
        REQUIRE(result);
    }

    {
        // Test with buffers of same size but different content
        std::vector<i32> data1 = {1, 2, 3};
        std::vector<i32> data2 = {1, 2, 4}; // Last element different

        sham::DeviceBuffer<i32> buf1(data1.size(), sched);
        sham::DeviceBuffer<i32> buf2(data2.size(), sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2);
        REQUIRE(!result);
    }

    {
        // Test error condition: cnt > buf1.size()
        constexpr u32 size    = 5;
        constexpr u32 cnt     = 10; // Greater than buffer size
        std::vector<i32> data = {1, 2, 3, 4, 5};

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        REQUIRE_EXCEPTION_THROW(
            shamalgs::primitives::equals(sched, buf1, buf2, cnt), std::invalid_argument);
    }

    {
        // Test error condition: cnt > buf2.size()
        constexpr u32 size1 = 10;
        constexpr u32 size2 = 5;
        constexpr u32 cnt   = 8; // Greater than buf2 size but less than buf1 size
        std::vector<i32> data1(size1, 1);
        std::vector<i32> data2(size2, 1);

        sham::DeviceBuffer<i32> buf1(size1, sched);
        sham::DeviceBuffer<i32> buf2(size2, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        REQUIRE_EXCEPTION_THROW(
            shamalgs::primitives::equals(sched, buf1, buf2, cnt), std::invalid_argument);
    }

    {
        // Test with u8 type
        constexpr u32 size    = 6;
        std::vector<u8> data1 = {1, 2, 3, 4, 5, 6};
        std::vector<u8> data2 = {1, 2, 3, 4, 5, 6};

        sham::DeviceBuffer<u8> buf1(size, sched);
        sham::DeviceBuffer<u8> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(result);
    }

    {
        // Test with u64 type
        constexpr u32 size     = 3;
        std::vector<u64> data1 = {1000000000ULL, 2000000000ULL, 3000000000ULL};
        std::vector<u64> data2 = {1000000000ULL, 2000000000ULL, 3000000001ULL}; // Last different

        sham::DeviceBuffer<u64> buf1(size, sched);
        sham::DeviceBuffer<u64> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(!result);
    }

    {
        // Test with f64 type
        constexpr u32 size    = 4;
        std::vector<f64> data = {3.14159, 2.71828, 1.41421, 1.73205};

        sham::DeviceBuffer<f64> buf1(size, sched);
        sham::DeviceBuffer<f64> buf2(size, sched);

        buf1.copy_from_stdvec(data);
        buf2.copy_from_stdvec(data);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, size);
        REQUIRE(result);
    }
    {
        // Test with zero count comparison
        constexpr u32 size     = 5;
        constexpr u32 cnt      = 0;
        std::vector<i32> data1 = {1, 2, 3, 4, 5};
        std::vector<i32> data2 = {6, 7, 8, 9, 10}; // Different data but cnt=0

        sham::DeviceBuffer<i32> buf1(size, sched);
        sham::DeviceBuffer<i32> buf2(size, sched);

        buf1.copy_from_stdvec(data1);
        buf2.copy_from_stdvec(data2);

        bool result = shamalgs::primitives::equals(sched, buf1, buf2, cnt);
        REQUIRE(result); // Should return true since no elements are compared
    }
}
