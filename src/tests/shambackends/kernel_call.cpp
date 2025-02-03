// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/kernel_call.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

TestStart(Unittest, "sham::kernel_call", testkernel_call, 1) {

    auto sched_ptr = shamsys::instance::get_compute_scheduler_ptr();
    auto queue     = sched_ptr->get_queue();

    using T = f64;
    sham::DeviceBuffer<T> buf{100, sched_ptr};

    buf.fill(1.0);

    sham::DeviceBuffer<T> buf2{100, sched_ptr};

    sham::kernel_call(
        queue, sham::MultiRef{buf}, sham::MultiRef{buf2}, 100, [](u32 i, const T *buf, T *buf2) {
            buf2[i] = buf[i];
        });

    std::vector res = buf2.copy_to_stdvec();

    for (u32 i = 0; i < 100; i++) {
        shamtest::asserts().assert_equal("check", res[i], 1.0);
    }
}

TestStart(Unittest, "sham:kernel_call_handle", testkernel_call_handle, 1) {

    auto sched_ptr = shamsys::instance::get_compute_scheduler_ptr();
    auto queue     = sched_ptr->get_queue();

    using T = f64;
    sham::DeviceBuffer<T> buf{128, sched_ptr};

    buf.fill(1.0);

    sham::DeviceBuffer<T> buf2{128, sched_ptr};

    u32 group_size = 16;
    sham::kernel_call_handle(
        queue,
        sham::MultiRef{buf},
        sham::MultiRef{buf2},
        group_size,
        128,
        [group_size](sycl::handler &cgh, const T *buf, T *buf2) {
            sycl::local_accessor<T> local_buf{16, cgh};

            return [=](u32 group_id, u32 local_id) {
                u32 i               = group_id * group_size + local_id;
                local_buf[local_id] = buf[i];
                buf2[i]             = local_buf[local_id];
            };
        });

    std::vector res = buf2.copy_to_stdvec();

    for (u32 i = 0; i < 128; i++) {
        shamtest::asserts().assert_equal("check", res[i], 1.0);
    }
}
