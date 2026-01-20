// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/gen_buffer_index.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shamalgs/primitives/gen_buffer_index", test_gen_buffer_index_primitive, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test gen_buffer_index with larger buffer
        constexpr u32 size = 100;
        auto buf           = shamalgs::primitives::gen_buffer_index(sched, size);

        REQUIRE_EQUAL(buf.get_size(), size);

        std::vector<u32> expected(size);
        for (u32 i = 0; i < size; ++i) {
            expected[i] = i;
        }
        std::vector<u32> result = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test gen_buffer_index with zero size
        constexpr u32 size = 0;
        auto buf           = shamalgs::primitives::gen_buffer_index(sched, size);

        REQUIRE_EQUAL(buf.get_size(), size);

        std::vector<u32> expected = {};
        std::vector<u32> result   = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }
}

TestStart(Unittest, "shamalgs/primitives/gen_buffer_index", test_fill_buffer_index_primitive, 1) {

    auto sched = shamsys::instance::get_compute_scheduler_ptr();

    {
        // Test fill_buffer_index with pre-allocated buffer
        constexpr u32 size = 10;
        sham::DeviceBuffer<u32> buf(size, sched);

        // Fill with some initial data to ensure it gets overwritten
        std::vector<u32> initial_data(size, 999);
        buf.copy_from_stdvec(initial_data);

        shamalgs::primitives::fill_buffer_index(buf, size);

        std::vector<u32> expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<u32> result   = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test fill_buffer_index with partial fill
        constexpr u32 buf_size  = 10;
        constexpr u32 fill_size = 5;
        sham::DeviceBuffer<u32> buf(buf_size, sched);

        // Initialize with some data
        std::vector<u32> initial_data(buf_size, 999);
        buf.copy_from_stdvec(initial_data);

        shamalgs::primitives::fill_buffer_index(buf, fill_size);

        std::vector<u32> expected = {0, 1, 2, 3, 4, 999, 999, 999, 999, 999};
        std::vector<u32> result   = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test fill_buffer_index with zero length
        constexpr u32 size = 5;
        sham::DeviceBuffer<u32> buf(size, sched);

        // Initialize with some data
        std::vector<u32> initial_data(size, 888);
        buf.copy_from_stdvec(initial_data);

        shamalgs::primitives::fill_buffer_index(buf, 0);

        // Should remain unchanged
        std::vector<u32> expected = {888, 888, 888, 888, 888};
        std::vector<u32> result   = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test error condition: fill_buffer_index with len > buf.get_size()
        constexpr u32 buf_size  = 5;
        constexpr u32 fill_size = 10;
        sham::DeviceBuffer<u32> buf(buf_size, sched);

        REQUIRE_EXCEPTION_THROW(
            shamalgs::primitives::fill_buffer_index(buf, fill_size), std::invalid_argument);
    }

    {
        // Test error condition: fill_buffer_index with len exactly equal to buf.get_size()
        constexpr u32 size = 7;
        sham::DeviceBuffer<u32> buf(size, sched);

        // This should not throw
        shamalgs::primitives::fill_buffer_index(buf, size);

        std::vector<u32> expected = {0, 1, 2, 3, 4, 5, 6};
        std::vector<u32> result   = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }

    {
        // Test gen_buffer_index produces same result as fill_buffer_index
        constexpr u32 size = 1000;
        auto buf           = shamalgs::primitives::gen_buffer_index(sched, size);

        REQUIRE_EQUAL(buf.get_size(), size);

        std::vector<u32> expected(size);
        for (u32 i = 0; i < size; ++i) {
            expected[i] = i;
        }
        std::vector<u32> result = buf.copy_to_stdvec();
        REQUIRE_EQUAL(result, expected);
    }
}
