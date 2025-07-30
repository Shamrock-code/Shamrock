// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/container/ResizableUSMBuffer.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>

TestStart(Unittest, "shamalgs/container/ResizableUSMBuffer", test_resizableUSMBuffer, 1) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Host);
        usm_rbuf.change_capacity(100);
    }

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Shared);
        usm_rbuf.change_capacity(100);
    }

    {
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Device);
        usm_rbuf.change_capacity(100);
    }

    { // test move constructor to check if no double free are performed

        shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Host);
        usm_rbuf.change_capacity(100);
        shamalgs::ResizableUSMBuffer<u32> usm_rbuf2{std::move(usm_rbuf)};
    }
}

TestStart(
    Unittest,
    "shamalgs/container/ResizableUSMBuffer:synchronisation",
    test_resizableUSMBuffer_sync,
    1) {

    sycl::queue &q = shamsys::instance::get_compute_queue();

    shamalgs::ResizableUSMBuffer<u32> usm_rbuf(q, shamalgs::Device);
    usm_rbuf.change_capacity(100);

    {
        std::vector<sycl::event> wait_list;
        u32 *acc = usm_rbuf.get_usm_ptr(wait_list);

        sycl::event ret = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            shambase::parallel_for(cgh, 100, "test1", [=](u32 id) {
                acc[id] = id;
            });
        });

        usm_rbuf.register_read_write_event(ret);
    }

    shamalgs::ResizableUSMBuffer<u32> usm_rbuf_sub1(q, shamalgs::Device);
    usm_rbuf_sub1.change_capacity(100);

    shamalgs::ResizableUSMBuffer<u32> usm_rbuf_sub2(q, shamalgs::Device);
    usm_rbuf_sub2.change_capacity(100);

    {
        std::vector<sycl::event> wait_list;
        const u32 *acc_src = usm_rbuf.get_usm_ptr_read_only(wait_list);
        u32 *acc_d1        = usm_rbuf_sub1.get_usm_ptr(wait_list);

        sycl::event ret = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            shambase::parallel_for(cgh, 100, "test2", [=](u32 id) {
                acc_d1[id] = acc_src[id] * 2;
            });
        });

        usm_rbuf.register_read_event(ret);
        usm_rbuf_sub1.register_read_write_event(ret);
    }

    {
        std::vector<sycl::event> wait_list;
        const u32 *acc_src = usm_rbuf.get_usm_ptr_read_only(wait_list);
        u32 *acc_d2        = usm_rbuf_sub2.get_usm_ptr(wait_list);

        sycl::event ret = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            shambase::parallel_for(cgh, 100, "test3", [=](u32 id) {
                acc_d2[id] = acc_src[id] * 3;
            });
        });

        usm_rbuf.register_read_event(ret);
        usm_rbuf_sub2.register_read_write_event(ret);
    }

    {
        std::vector<sycl::event> wait_list;
        const u32 *acc_src1 = usm_rbuf_sub1.get_usm_ptr_read_only(wait_list);
        const u32 *acc_src2 = usm_rbuf_sub2.get_usm_ptr_read_only(wait_list);
        u32 *acc_d          = usm_rbuf.get_usm_ptr(wait_list);

        sycl::event ret = q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(wait_list);

            shambase::parallel_for(cgh, 100, "test4", [=](u32 id) {
                acc_d[id] = acc_src1[id] + acc_src2[id];
            });
        });

        usm_rbuf.register_read_write_event(ret);
        usm_rbuf_sub1.register_read_event(ret);
        usm_rbuf_sub2.register_read_event(ret);
    }
}
