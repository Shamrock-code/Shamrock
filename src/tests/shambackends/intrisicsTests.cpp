// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/intrinsics.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <set>

TestStart(Unittest, "shambackends/intrisics::get_sm_id", test_get_sm_id, 1) {

    const size_t sz = 256 * 256;
    using T         = i32;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::DeviceBuffer<T> buf{sz, dev_sched};

    sham::EventList deps;
    auto ptr = buf.get_write_access(deps);

    auto e = q.submit(deps, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
            u64 gid = id.get_linear_id();
#ifdef SHAMROCK_INTRISICS_GET_SMID_AVAILABLE
            ptr[gid] = (i32) sham::get_sm_id();
#else
            ptr[gid] = -1;
#endif
        });
    });

    buf.complete_event_state(e);

    auto ret = buf.copy_to_stdvec();

    std::set<i32> sm_ids(ret.begin(), ret.end());

    if (sm_ids == std::set<i32>{-1}) {
        logger::warn_ln("Test", "sham::get_sm_id() is not available on this device");
    } else {
        logger::raw_ln(sm_ids);
    }
}

TestStart(Unittest, "shambackends/intrisics::get_device_clock", test_get_device_clock, 1) {

    const size_t sz = 256 * 256;
    using T         = i64;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::DeviceBuffer<T> buf{sz, dev_sched};

    sham::EventList deps;
    auto ptr = buf.get_write_access(deps);

    auto e = q.submit(deps, [&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{sz}, [=](sycl::item<1> id) {
            u64 gid = id.get_linear_id();
#ifdef SHAMROCK_INTRISICS_GET_DEVICE_CLOCK_AVAILABLE
            ptr[gid] = (i64) sham::get_device_clock();
#else
            ptr[gid] = -1;
#endif
        });
    });

    buf.complete_event_state(e);

    auto ret = buf.copy_to_stdvec();

    std::set<i64> sm_ids(ret.begin(), ret.end());

    if (sm_ids == std::set<i64>{-1}) {
        logger::warn_ln("Test", "sham::get_device_clock() is not available on this device");
    } else {
        auto first = *sm_ids.begin();
        auto last  = *sm_ids.rbegin();

        auto comp = [](i64 a, i64 b) {
            return (b - a) < 1e8;
        };

        REQUIRE_EQUAL_CUSTOM_COMP(first, last, comp);
    }
}
