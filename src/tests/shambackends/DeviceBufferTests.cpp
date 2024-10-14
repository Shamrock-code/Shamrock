// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <vector>

void fill(sham::DeviceQueue &q, sham::DeviceBuffer<int> &f_a, int value) {

    sham::EventList depends_list;

    int *a = f_a.get_write_access(depends_list);

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            a[indx] = value;
        });
    });

    f_a.complete_event_state(e);
}

void add(
    sham::DeviceQueue &q,
    sham::DeviceBuffer<int> &f_a,
    sham::DeviceBuffer<int> &f_b,
    sham::DeviceBuffer<int> &f_c) {

    sham::EventList depends_list;

    const int *a = f_a.get_read_access(depends_list);
    const int *b = f_b.get_read_access(depends_list);
    int *c       = f_c.get_write_access(depends_list);

    sycl::event e = q.submit(depends_list, [&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            c[indx] = b[indx] + a[indx];
        });
    });

    f_a.complete_event_state(e);
    f_b.complete_event_state(e);
    f_c.complete_event_state(e);
}

TestStart(Unittest, "shambackends/DeviceBuffer:smalltaskgraph", DeviceBuffer_small_task_graph, 1) {

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> a{1000, dev_sched};
    sham::DeviceBuffer<int> b{1000, dev_sched};
    sham::DeviceBuffer<int> c{1000, dev_sched};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    fill(q, a, 77);
    fill(q, b, 33);

    add(q, a, b, c);
}

TestStart(Unittest, "shambackends/DeviceBuffer:resize", DeviceBuffer_resize, 1) {

    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> a{1000, dev_sched};
    a.fill(77);

    a.resize(2000);

    REQUIRE(a.get_size() == 2000);
    REQUIRE(a.get_mem_usage() >= a.to_bytesize(2000));

    {
        std::vector<int> b = a.copy_to_stdvec();
        REQUIRE(b.size() == 2000);

        for (int i = 0; i < 1000; i++) {
            REQUIRE(b[i] == 77);
        }
    }
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill", DeviceBuffer_fill1, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Fill the buffer with value 5
    buffer.fill(5);

    {
        std::vector<int> b = buffer.copy_to_stdvec();
        REQUIRE(b.size() == 10);

        // Check that the buffer is filled with the correct value
        for (int i = 0; i < 10; i++) {
            REQUIRE(b[i] == 5);
        }
    }
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill(with count)", DeviceBuffer_fill2, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Fill the buffer with value 5
    buffer.fill(0);

    // Fill the buffer with value 5, starting from index 2, with count 5
    buffer.fill(5, 2, 5);

    {
        std::vector<int> b = buffer.copy_to_stdvec();
        REQUIRE(b.size() == 10);

        // Check that the buffer is filled with the correct value
        for (int i = 0; i < 10; i++) {
            if (i >= 2 && i < 7) {
                _AssertEqual(b[i], 5);
            } else {
                _AssertEqual(b[i], 0);
            }
        }
    }
}

TestStart(Unittest, "shambackends/DeviceBuffer:fill(exception)", DeviceBuffer_fill3, 1) {
    std::shared_ptr<sham::DeviceScheduler> dev_sched
        = shamsys::instance::get_compute_scheduler_ptr();

    // Create a device buffer with size 10
    sham::DeviceBuffer<int> buffer(10, dev_sched);

    // Try to fill the buffer with value 5, starting from index 15, with count 5
    _Assert_throw(buffer.fill(5, 15, 5), std::invalid_argument);
}
