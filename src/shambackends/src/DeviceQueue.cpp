// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceQueue.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceQueue.hpp"
#include <utility>

namespace sham {

    auto build_queue = [](sycl::context &ctx, sycl::device &dev, bool in_order) -> sycl::queue {
        if (in_order) {
            return sycl::queue{ctx, dev, sycl::property::queue::in_order{}};
        } else {
            return sycl::queue{ctx, dev};
        }
    };

    DeviceQueue::DeviceQueue(
        std::string queue_name, std::shared_ptr<DeviceContext> _ctx, bool in_order)
        : queue_name(std::move(queue_name)), ctx(std::move(_ctx)), in_order(in_order),
          q(build_queue(ctx->ctx, ctx->device->dev, in_order)) {}

    void DeviceQueue::test() {
        auto test_kernel = [](sycl::queue &q) {
            sycl::buffer<u32> b(10);

            q.submit([&](sycl::handler &cgh) {
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
