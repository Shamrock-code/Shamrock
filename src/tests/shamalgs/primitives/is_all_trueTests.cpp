// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/logs/loglevels.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

TestStart(
    Unittest, "shamalgs/primitives/is_all_true:sycl_buffer", test_is_all_true_sycl_buffer, 1) {

    {
        // Test with all true values
        constexpr u32 size = 10;
        std::vector<u8> data(size, 1);
        sycl::buffer<u8> buf(data);

        bool result = shamalgs::primitives::is_all_true(buf, size);
        REQUIRE(result);
    }

    {
        // Test with all false values
        constexpr u32 size = 10;
        std::vector<u8> data(size, 0);
        sycl::buffer<u8> buf(data);

        bool result = shamalgs::primitives::is_all_true(buf, size);
        REQUIRE(!result);
    }

    {
        // Test with mixed values (should return false)
        constexpr u32 size   = 5;
        std::vector<u8> data = {1, 0, 1, 1, 0};
        sycl::buffer<u8> buf(data);

        bool result = shamalgs::primitives::is_all_true(buf, size);
        REQUIRE(!result);
    }

    {
        // Test with single true value
        constexpr u32 size   = 1;
        std::vector<u8> data = {1};
        sycl::buffer<u8> buf(data);

        bool result = shamalgs::primitives::is_all_true(buf, size);
        REQUIRE(result);
    }

    {
        // Test with single false value
        constexpr u32 size   = 1;
        std::vector<u8> data = {0};
        sycl::buffer<u8> buf(data);

        bool result = shamalgs::primitives::is_all_true(buf, size);
        REQUIRE(!result);
    }
}

TestStart(Unittest, "shamalgs/primitives/is_all_true:USM", test_is_all_true_device_buffer, 1) {

    auto test_impl = [&]() {
        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        {
            // Test with all true values
            constexpr u32 size = 10;
            std::vector<u8> data(size, 1);
            sham::DeviceBuffer<u8> buf(size, sched);
            buf.copy_from_stdvec(data);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(result);
        }

        {
            // Test with all false values
            constexpr u32 size = 10;
            std::vector<u8> data(size, 0);
            sham::DeviceBuffer<u8> buf(size, sched);
            buf.copy_from_stdvec(data);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(!result);
        }

        {
            // Test with mixed values (should return false)
            constexpr u32 size   = 5;
            std::vector<u8> data = {1, 0, 1, 1, 0};
            sham::DeviceBuffer<u8> buf(size, sched);
            buf.copy_from_stdvec(data);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(!result);
        }

        {
            // Test with empty buffer
            constexpr u32 size = 0;
            sham::DeviceBuffer<u8> buf(size, sched);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(result); // Empty buffer should return true
        }

        {
            // Test with single true value
            constexpr u32 size   = 1;
            std::vector<u8> data = {1};
            sham::DeviceBuffer<u8> buf(size, sched);
            buf.copy_from_stdvec(data);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(result);
        }

        {
            // Test with single false value
            constexpr u32 size   = 1;
            std::vector<u8> data = {0};
            sham::DeviceBuffer<u8> buf(size, sched);
            buf.copy_from_stdvec(data);

            bool result = shamalgs::primitives::is_all_true(buf, size);
            REQUIRE(!result);
        }
    };

    std::vector<std::string> impls = shamalgs::primitives::impl::get_impl_list_is_all_true();
    for (auto &impl : impls) {
        shamalgs::primitives::impl::set_impl_is_all_true(impl);
        shamlog_info_ln("tests", "testing implementation:", impl);
        test_impl();
    }
}
