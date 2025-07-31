// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

TestStart(Unittest, "shambackends/DeviceBuffer", DeviceBuffer_consttructor, 1) {
    using namespace sham;

    constexpr USMKindTarget target = USMKindTarget::device;
    const size_t sz                = 10;
    using T                        = int;

    std::shared_ptr<DeviceScheduler> sched = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceBuffer<T, target> buf{sz, sched};

    sham::EventList e;

    REQUIRE(buf.get_read_access(e) != nullptr);
    buf.complete_event_state(sycl::event{});
    REQUIRE(buf.get_write_access(e) != nullptr);
    buf.complete_event_state(sycl::event{});
    REQUIRE(buf.get_size() == sz);
    REQUIRE(buf.get_mem_usage() >= sz * sizeof(T));

    e.wait_and_throw();

    // REQUIRE(buf.get_sched() == sched);
}
