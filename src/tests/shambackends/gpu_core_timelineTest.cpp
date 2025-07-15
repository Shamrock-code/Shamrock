// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/gpu_core_timeline.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/EventList.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"

TestStart(
    Unittest, "shambackends/gpu_core_timeline_profilier", gpu_core_timeline_profilier_test, 1) {

    const size_t sz = 256 * 256;
    using T         = int;

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<T> buf{sz, dev_sched};
    buf.fill(10);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    sham::gpu_core_timeline_profilier profiler(
        shamsys::instance::get_compute_scheduler_ptr(), 1000000);
    profiler.setFrameStartClock();

    sham::EventList deps;
    auto ptr            = buf.get_write_access(deps);
    auto gpu_core_timer = profiler.get_write_access(deps);

    auto e = q.submit(deps, [&](sycl::handler &cgh) {
        sham::gpu_core_timeline_profilier::local_access_t gpu_core_timer_data(cgh);

        u64 group_size = 64;
        cgh.parallel_for(shambase::make_range(sz, group_size), [=](sycl::nd_item<1> id) {
            u64 gid = id.get_global_linear_id();
            if (gid >= sz)
                return;

            gpu_core_timer.init_timeline_event(id, gpu_core_timer_data);

            gpu_core_timer.start_timeline_event(gpu_core_timer_data);
            u32 id_a = (u32) gid;

            // force something very unbalanced
            for (u32 i = 0; i < 8 * (id_a % (group_size * 2)); i++) {
                ptr[gid] = static_cast<T>(sycl::sqrt(f64(ptr[gid])));
            }

            gpu_core_timer.end_timeline_event(gpu_core_timer_data);
        });
    });

    buf.complete_event_state(e);
    profiler.complete_event_state(e);

    profiler.dump_to_file("gpu_core_timeline_profilier_test.json");
    // profiler.open_file("gpu_core_timeline_profilier_test.json");
}
