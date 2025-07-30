// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceQueue.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceQueue.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/logs.hpp"
#include <utility>

std::string SHAMROCK_WAIT_AFTER_SUBMIT = shamcmdopt::getenv_str_default_register(
    "SHAMROCK_WAIT_AFTER_SUBMIT", "0", "Make queues wait after submit (default: 0)");

namespace sham {

    auto build_queue = [](sycl::context &ctx, sycl::device &dev, bool in_order) -> sycl::queue {
        if (in_order) {
            return sycl::queue{ctx, dev, sycl::property::queue::in_order{}};
        } else {
            return sycl::queue{ctx, dev};
        }
    };

    auto parse_wait_after_submit = []() -> bool {
        bool ret = SHAMROCK_WAIT_AFTER_SUBMIT == "1";

        if (ret) {
            shamcomm::logs::warn_ln("Backends", "DeviceQueue :", "wait_after_submit is on !");
        }

        return ret;
    };

    bool env_var_wait_after_submit_set = parse_wait_after_submit();

    DeviceQueue::DeviceQueue(
        std::string queue_name, std::shared_ptr<DeviceContext> _ctx, bool in_order)
        : queue_name(std::move(queue_name)), ctx(std::move(_ctx)), in_order(in_order),
          q(build_queue(ctx->ctx, ctx->device->dev, in_order)),
          wait_after_submit(env_var_wait_after_submit_set) {}

    void DeviceQueue::test() {
        auto test_kernel = [&](sycl::queue &q) {
            sycl::buffer<u32> b(10);

            submit([&b](sycl::handler &cgh) {
                sycl::accessor acc{b, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range<1>{10}, [=](sycl::item<1> i) {
                    acc[i] = i.get_linear_id();
                });
            });

            q.wait();

            {
                sycl::host_accessor acc{b, sycl::read_only};
                if (acc[9] != 9) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "The chosen SYCL queue cannot execute a basic kernel");
                }
            }
        };

        std::exception_ptr eptr;
        try {
            test_kernel(q);
            // logger::info_ln("NodeInstance", "selected queue
            // :",q.get_device().get_info<sycl::info::device::name>()," working !");
        } catch (...) {
            eptr = std::current_exception(); // capture
        }

        if (eptr) {
            // logger::err_ln("NodeInstance", "selected queue
            // :",q.get_device().get_info<sycl::info::device::name>(),"does not function properly");
            std::rethrow_exception(eptr);
        }
    }

} // namespace sham
