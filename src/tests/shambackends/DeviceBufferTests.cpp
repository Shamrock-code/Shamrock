// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/USMPtrHolder.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include <vector>


void fill(sycl::queue & q, sham::DeviceBuffer<int> & f_a, int value){

    std::vector<sycl::event> depends_list {};
    int* a = f_a.get_write_access(depends_list);

    sycl::event e = q.submit([&](sycl::handler& h) {

        h.depends_on(depends_list);

        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            a[indx] = value;
        });

    });

    f_a.complete_event_state(e);
}

void add(sycl::queue & q,sham::DeviceBuffer<int> & f_a,sham::DeviceBuffer<int> & f_b,sham::DeviceBuffer<int> & f_c)
{
    
    std::vector<sycl::event> depends_list {};

    const int* a = f_a.get_read_access(depends_list);
    const int* b = f_b.get_read_access(depends_list);
    int* c = f_c.get_write_access(depends_list);

    sycl::event e = q.submit([&](sycl::handler& h) {

        h.depends_on(depends_list);

        h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> indx) {
            c[indx] = b[indx] + a[indx];
        });

    });

    f_a.complete_event_state(e);
    f_b.complete_event_state(e);
    f_c.complete_event_state(e);

}

TestStart(Unittest, "shambackends/DeviceBuffer:smalltaskgraph", DeviceBuffer_small_task_graph, 1){

    std::shared_ptr<sham::DeviceScheduler> dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    sham::DeviceBuffer<int> a{1000,dev_sched};
    sham::DeviceBuffer<int> b{1000,dev_sched};
    sham::DeviceBuffer<int> c{1000,dev_sched};

    sycl::queue & q = shamsys::instance::get_compute_scheduler().get_queue().q;

    fill(q,a,77);
    fill(q,b,33);

    add(q, a, b, c);

}