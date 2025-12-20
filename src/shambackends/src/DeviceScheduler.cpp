// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DeviceScheduler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceScheduler.hpp"
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"

namespace sham {

    DeviceScheduler::DeviceScheduler(std::shared_ptr<DeviceContext> ctx) : ctx(ctx) {

        queues.push_back(std::make_unique<DeviceQueue>("main_queue", ctx, false));
    }

    DeviceQueue &DeviceScheduler::get_queue(u32 i) { return *(queues.at(i)); }

    void DeviceScheduler::print_info() {

        ctx->print_info();

        shamcomm::logs::raw_ln("  Queue list:");
        for (auto &q : queues) {
            std::string tmp
                = shambase::format("   - name : {:20s} in order : {}", q->queue_name, q->in_order);
            shamcomm::logs::raw_ln(tmp);
        }
    }

    bool DeviceScheduler::use_direct_comm() { return ctx->use_direct_comm(); }

    void test_device_scheduler(sham::DeviceScheduler_ptr dev_sched) {
        for (auto &q : dev_sched->queues) {

            auto test_kernel = [&](sham::DeviceQueue &q) {
                sham::DeviceBuffer<u32> b(10, dev_sched);

                sham::kernel_call(q, sham::MultiRef{}, sham::MultiRef{b}, 10, [](u32 i, u32 *b) {
                    b[i] = i;
                });

                {
                    std::vector<u32> acc = b.copy_to_stdvec();
                    if (acc[9] != 9) {
                        throw shambase::make_except_with_loc<std::runtime_error>(
                            "The chosen SYCL queue cannot execute a basic kernel");
                    }
                }
            };

            std::exception_ptr eptr;
            try {
                sham::DeviceQueue &qref     = shambase::get_check_ref(q);
                sham::DeviceContext &ctxref = shambase::get_check_ref(qref.ctx);
                sham::Device &deviceref     = shambase::get_check_ref(ctxref.device);
                std::string device_name     = deviceref.dev.get_info<sycl::info::device::name>();

                logger::debug_ln("Backends", "[Queue testing] name = ", device_name);
                test_kernel(qref);
                logger::debug_ln(
                    "Backends", "[Queue testing] name = ", device_name, " -> working !");

            } catch (...) {
                eptr = std::current_exception(); // capture
            }

            if (eptr) {
                sham::DeviceQueue &qref     = shambase::get_check_ref(q);
                sham::DeviceContext &ctxref = shambase::get_check_ref(qref.ctx);
                sham::Device &deviceref     = shambase::get_check_ref(ctxref.device);
                std::string device_name     = deviceref.dev.get_info<sycl::info::device::name>();
                logger::err_ln(
                    "Backends", "[Queue testing] name = ", device_name, " -> not working !");
                std::rethrow_exception(eptr);
            }
        }
    }
} // namespace sham
