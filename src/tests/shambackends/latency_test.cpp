// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shambackends/sycl.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

f64 test_buffer_out_of_order(u32 buf_size, u32 stream_count, u32 repeat_count) {
    StackEntry stack_loc{};

    std::vector<std::unique_ptr<sycl::buffer<f64>>> bufs{};

    // allocate and init
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        std::unique_ptr<sycl::buffer<f64>> buf = std::make_unique<sycl::buffer<f64>>(buf_size);

        shamsys::instance::get_compute_queue()
            .submit([&](sycl::handler &cgh) {
                sycl::accessor acc{*buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    acc[gid] = gid;
                });
            })
            .wait();

        bufs.push_back(std::move(buf));
    }

    shambase::Timer timer;
    timer.start();

    for (u32 irepeat = 0; irepeat < repeat_count; irepeat++) {
        for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc{*bufs[ibuf], cgh, sycl::read_write};

                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    acc[gid] = acc[gid] * f64(1.1);
                });
            });
        }
    }

    shamsys::instance::get_compute_queue().wait();

    timer.end();
    return timer.elasped_sec();
}

f64 test_buffer_in_order(u32 buf_size, u32 stream_count, u32 repeat_count) {
    StackEntry stack_loc{};

    sycl::queue q = sycl::queue{
        shamsys::instance::get_compute_scheduler().ctx->ctx,
        shamsys::instance::get_compute_scheduler().ctx->device->dev,
        sycl::property::queue::in_order{}};

    std::vector<std::unique_ptr<sycl::buffer<f64>>> bufs{};

    // allocate and init
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        std::unique_ptr<sycl::buffer<f64>> buf = std::make_unique<sycl::buffer<f64>>(buf_size);

        q.submit([&](sycl::handler &cgh) {
             sycl::accessor acc{*buf, cgh, sycl::write_only, sycl::no_init};

             cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                 u32 gid = id.get_linear_id();

                 acc[gid] = gid;
             });
         }).wait();

        bufs.push_back(std::move(buf));
    }

    shambase::Timer timer;
    timer.start();

    for (u32 irepeat = 0; irepeat < repeat_count; irepeat++) {
        for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
            q.submit([&](sycl::handler &cgh) {
                sycl::accessor acc{*bufs[ibuf], cgh, sycl::read_write};

                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    acc[gid] = acc[gid] * f64(1.1);
                });
            });
        }
    }

    q.wait();

    timer.end();
    return timer.elasped_sec();
}

f64 test_buffer_in_order_multi_queue(u32 buf_size, u32 stream_count, u32 repeat_count) {
    StackEntry stack_loc{};

    std::vector<std::unique_ptr<sycl::queue>> queues{};
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        queues.push_back(
            std::make_unique<sycl::queue>(sycl::queue{
                shamsys::instance::get_compute_scheduler().ctx->ctx,
                shamsys::instance::get_compute_scheduler().ctx->device->dev,
                sycl::property::queue::in_order{}}));
    }

    std::vector<std::unique_ptr<sycl::buffer<f64>>> bufs{};

    // allocate and init
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        std::unique_ptr<sycl::buffer<f64>> buf = std::make_unique<sycl::buffer<f64>>(buf_size);

        queues[ibuf]
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor acc{*buf, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    acc[gid] = gid;
                });
            })
            .wait();

        bufs.push_back(std::move(buf));
    }

    shambase::Timer timer;
    timer.start();

    for (u32 irepeat = 0; irepeat < repeat_count; irepeat++) {
        for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
            queues[ibuf]->submit([&](sycl::handler &cgh) {
                sycl::accessor acc{*bufs[ibuf], cgh, sycl::read_write};

                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    acc[gid] = acc[gid] * f64(1.1);
                });
            });
        }
    }

    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        queues[ibuf]->wait();
    }
    timer.end();
    return timer.elasped_sec();
}

f64 test_usm_in_order(u32 buf_size, u32 stream_count, u32 repeat_count) {

    StackEntry stack_loc{};

    sycl::queue q = sycl::queue{
        shamsys::instance::get_compute_scheduler().ctx->ctx,
        shamsys::instance::get_compute_scheduler().ctx->device->dev,
        sycl::property::queue::in_order{}};

    std::vector<f64 *> bufs{};

    // allocate and init
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        f64 *buf = sycl::malloc_device<f64>(buf_size, q);

        q.submit([&](sycl::handler &cgh) {
             cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                 u32 gid = id.get_linear_id();

                 buf[gid] = gid;
             });
         }).wait();

        bufs.push_back(std::move(buf));
    }

    shambase::Timer timer;
    timer.start();

    for (u32 irepeat = 0; irepeat < repeat_count; irepeat++) {
        for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
            f64 *buf = bufs[ibuf];
            q.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    buf[gid] = buf[gid] * f64(1.1);
                });
            });
        }
    }

    q.wait();
    timer.end();

    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        f64 *buf = bufs[ibuf];
        sycl::free(buf, q);
    }
    return timer.elasped_sec();
}

f64 test_usm_in_order_multi_queue(u32 buf_size, u32 stream_count, u32 repeat_count) {

    StackEntry stack_loc{};

    std::vector<std::unique_ptr<sycl::queue>> queues{};
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        queues.push_back(
            std::make_unique<sycl::queue>(sycl::queue{
                shamsys::instance::get_compute_scheduler().ctx->ctx,
                shamsys::instance::get_compute_scheduler().ctx->device->dev,
                sycl::property::queue::in_order{}}));
    }

    std::vector<f64 *> bufs{};

    // allocate and init
    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        f64 *buf = sycl::malloc_device<f64>(buf_size, *queues[ibuf]);

        queues[ibuf]
            ->submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    buf[gid] = gid;
                });
            })
            .wait();

        bufs.push_back(std::move(buf));
    }

    shambase::Timer timer;
    timer.start();

    for (u32 irepeat = 0; irepeat < repeat_count; irepeat++) {
        for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
            f64 *buf = bufs[ibuf];
            queues[ibuf]->submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{buf_size}, [=](sycl::item<1> id) {
                    u32 gid = id.get_linear_id();

                    buf[gid] = buf[gid] * f64(1.1);
                });
            });
        }
    }

    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        queues[ibuf]->wait();
    }
    timer.end();

    for (u32 ibuf = 0; ibuf < stream_count; ibuf++) {
        f64 *buf = bufs[ibuf];
        sycl::free(buf, *queues[ibuf]);
    }
    return timer.elasped_sec();
}

TestStart(Benchmark, "latency_sycl", latency_sycl, 1) {
    shamcomm::logs::raw_ln(test_buffer_out_of_order(1e4, 10, 1000));
    shamcomm::logs::raw_ln(test_buffer_in_order(1e4, 10, 1000));
    shamcomm::logs::raw_ln(test_buffer_in_order_multi_queue(1e4, 10, 1000));
    shamcomm::logs::raw_ln(test_usm_in_order(1e4, 10, 1000));
    shamcomm::logs::raw_ln(test_usm_in_order_multi_queue(1e4, 10, 1000));
}
