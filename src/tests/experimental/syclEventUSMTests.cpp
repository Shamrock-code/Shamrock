// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/shamtest.hpp"

struct DeviceQueue {

    sycl::queue q;

    void wait() { q.wait(); }

    void wait_and_throw() { q.wait(); }

    DeviceQueue &operator=(const DeviceQueue &) = delete;
    DeviceQueue(const DeviceQueue &)            = delete;
    DeviceQueue &operator=(DeviceQueue &&)      = delete;
    DeviceQueue(DeviceQueue &&)                 = delete;

    operator const sycl::queue &() const { return q; }
};

struct QueueEvent {

    sycl::event e;

    QueueEvent() : e{} {}
    QueueEvent(sycl::event &&event) : e{std::forward<sycl::event>(event)} {}

    ~QueueEvent() = default;

    void wait() { e.wait(); }

    operator sycl::event() const { return e; }
};

enum BufferType { HOST, Device, Shared };

template<class T>
class ResizableUSMBuffer {
    T *usm_ptr = nullptr;

    u32 buf_size;
    u32 val_count;

    BufferType type;

    inline void alloc(DeviceQueue &q) {
        if (usm_ptr != nullptr) {
            throw;
        }

        if (type == HOST) {
            usm_ptr = sycl::malloc_host<T>(buf_size, q);
        } else if (type == Device) {
            usm_ptr = sycl::malloc_device<T>(buf_size, q);
        } else if (type == Shared) {
            usm_ptr = sycl::malloc_shared<T>(buf_size, q);
        }
    }

    inline void free(DeviceQueue &q) {
        sycl::free(usm_ptr, q);
        usm_ptr = nullptr;
    }
};

template class ResizableUSMBuffer<u32>;

class ker1test;
class ker2test;

QueueEvent sub_func(QueueEvent &e, u32 *ptr) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    QueueEvent e1 = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(e);

        cgh.memset(ptr, 0, 100);
    });

    QueueEvent e2 = q.submit([&, ptr](sycl::handler &cgh) {
        cgh.depends_on(e1);

        cgh.parallel_for<ker1test>(sycl::range<1>{100}, [=](sycl::id<1> idx) {
            // Initialize each buffer element with its own rank number starting at 0
            ptr[idx] = idx;
        }); // End of the kernel function
    });

    shamlog_debug_ln("TEST", "return sub_func");

    return e2;
}

TestStart(ValidationTest, "test-usm-event-arch", usm_event_test, 1) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    shamlog_debug_ln("TEST", "alloc 1");
    u32 *ptr1 = sycl::malloc_device<u32>(100, q);

    shamlog_debug_ln("TEST", "alloc 2");
    u32 *ptr2 = sycl::malloc_device<u32>(100, q);

    QueueEvent e1, e2;

    shamlog_debug_ln("TEST", "sub_func 1");
    QueueEvent e3 = sub_func(e1, ptr1);
    shamlog_debug_ln("TEST", "sub_func 2");
    QueueEvent e4 = sub_func(e2, ptr2);

    shamlog_debug_ln("TEST", "alloc 3");
    u32 *ptr3 = sycl::malloc_device<u32>(100, q);

    shamlog_debug_ln("TEST", "ker2");
    QueueEvent e5 = q.submit([&, ptr1, ptr2, ptr3](sycl::handler &cgh) {
        cgh.depends_on({e3, e4});

        cgh.parallel_for<ker2test>(sycl::range<1>{100}, [=](sycl::id<1> idx) {
            ptr3[idx] = ptr1[idx] + ptr2[idx];
        });
    });

    shamlog_debug_ln("TEST", "recover");
    std::array<u32, 100> recov;
    q.submit([&, ptr3](sycl::handler &cgh) {
         cgh.depends_on(e5);

         cgh.memcpy(recov.data(), ptr3, sizeof(u32) * 100);
     }).wait();
}
