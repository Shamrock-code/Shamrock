// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"

namespace {
    void test_queue(const sham::DeviceScheduler_ptr &dev_sched, sham::DeviceQueue &q) {
        sham::DeviceBuffer<u32> b(10, dev_sched);

        sham::kernel_call(q, sham::MultiRef{}, sham::MultiRef{b}, 10, [](u32 i, u32 *__restrict b) {
            b[i] = i;
        });

        std::vector<u32> expected_acc = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

        std::vector<u32> acc = b.copy_to_stdvec();
        if (acc != expected_acc) {
            auto &ctx    = shambase::get_check_ref(q.ctx);
            auto &device = shambase::get_check_ref(ctx.device);
            throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                "The chosen SYCL queue (name={}, device={}) cannot execute a basic kernel\n"
                "  expected acc = {}\n"
                "  actual acc = {}",
                q.queue_name,
                device.dev.get_info<sycl::info::device::name>(),
                expected_acc,
                acc));
        }
    }
} // namespace

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

            sham::DeviceQueue &qref     = shambase::get_check_ref(q);
            sham::DeviceContext &ctxref = shambase::get_check_ref(qref.ctx);
            sham::Device &deviceref     = shambase::get_check_ref(ctxref.device);
            std::string device_name     = deviceref.dev.get_info<sycl::info::device::name>();

            std::exception_ptr eptr;
            try {
                logger::debug_ln("Backends", "[Queue testing] name = ", device_name);
                test_queue(dev_sched, qref);
                logger::debug_ln(
                    "Backends", "[Queue testing] name = ", device_name, " -> working !");
            } catch (...) {
                eptr = std::current_exception(); // capture
            }

            if (eptr) {
                logger::err_ln(
                    "Backends", "[Queue testing] name = ", device_name, " -> not working !");
                std::rethrow_exception(eptr);
            }
        }

        logger::debug_ln("Backends", "[Alloc testing] starting...");

        sham::DeviceQueue &qref     = shambase::get_check_ref(dev_sched).get_queue();
        sham::DeviceContext &ctxref = shambase::get_check_ref(qref.ctx);

        USMPtrHolder<sham::device> ptr1024_dev
            = USMPtrHolder<sham::device>::create(1024, dev_sched, 8);
        ptr1024_dev.free_ptr();
        USMPtrHolder<sham::host> ptr1024_host
            = USMPtrHolder<sham::host>::create(1024, dev_sched, 8);
        ptr1024_host.free_ptr();

        auto &dev = shambase::get_check_ref(ctxref.device);

        size_t GBval = 1024 * 1024 * 1024;
        // avoid <8GB card, they won't run at that scale anyway
        if (dev.prop.global_mem_size > usize(3 * GBval)) {

            if (dev.prop.max_mem_alloc_size_dev > 2 * GBval) {
                try {
                    USMPtrHolder<sham::device> ptr2G_dev
                        = USMPtrHolder<sham::device>::create(2 * GBval + 1024, dev_sched, 8);
                    ptr2G_dev.free_ptr();
                } catch (std::runtime_error &e) {
                    logger::warn_ln(
                        "Backends",
                        " name = ",
                        dev.dev.get_info<sycl::info::device::name>(),
                        " -> large device allocation (>2GB) not working !");
                    dev.prop.max_mem_alloc_size_dev = i32_max;
                }
            }

            if (dev.prop.max_mem_alloc_size_host > 2 * GBval) {
                try {
                    USMPtrHolder<sham::host> ptr2G_host
                        = USMPtrHolder<sham::host>::create(2 * GBval + 1024, dev_sched, 8);
                    ptr2G_host.free_ptr();
                } catch (std::runtime_error &e) {
                    logger::warn_ln(
                        "Backends",
                        " name = ",
                        dev.dev.get_info<sycl::info::device::name>(),
                        " -> large host allocation (>2GB) not working !");
                    dev.prop.max_mem_alloc_size_host = i32_max;
                }
            }
        }

        logger::debug_ln("Backends", "[Alloc testing] done !");
    }
} // namespace sham
