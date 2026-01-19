// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file internal_alloc.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the methods to actually allocate memory
 */

#include "shambase/memory.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/string.hpp"
#include "shambackends/details/internal_alloc.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <exception>

namespace {

    sham::MemPerfInfos mem_perf_infos;

    void register_alloc_device(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_device += size;
        mem_perf_infos.time_alloc_device += timed;
        mem_perf_infos.max_allocated_byte_device = std::max(
            mem_perf_infos.max_allocated_byte_device, mem_perf_infos.allocated_byte_device);

        shambase::profiling::register_counter_val(
            "Device Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_device);
        shambase::profiling::register_counter_val(
            "Device alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_device);
    }

    void register_alloc_shared(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_shared += size;
        mem_perf_infos.time_alloc_shared += timed;
        mem_perf_infos.max_allocated_byte_shared = std::max(
            mem_perf_infos.max_allocated_byte_shared, mem_perf_infos.allocated_byte_shared);

        shambase::profiling::register_counter_val(

            "Shared Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_shared);
        shambase::profiling::register_counter_val(
            "Shared alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_shared);
    }

    void register_alloc_host(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_host += size;
        mem_perf_infos.time_alloc_host += timed;
        mem_perf_infos.max_allocated_byte_host
            = std::max(mem_perf_infos.max_allocated_byte_host, mem_perf_infos.allocated_byte_host);

        shambase::profiling::register_counter_val(
            "Host Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_host);
        shambase::profiling::register_counter_val(
            "Host alloc time", shambase::details::get_wtime(), mem_perf_infos.time_alloc_host);
    }

    void register_free_device(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_device -= size;
        mem_perf_infos.time_free_device += timed;

        shambase::profiling::register_counter_val(
            "Device Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_device);
        shambase::profiling::register_counter_val(
            "Device free time", shambase::details::get_wtime(), mem_perf_infos.time_free_device);
    }

    void register_free_shared(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_shared -= size;
        mem_perf_infos.time_free_shared += timed;

        shambase::profiling::register_counter_val(
            "Shared Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_shared);
        shambase::profiling::register_counter_val(
            "Shared free time", shambase::details::get_wtime(), mem_perf_infos.time_free_shared);
    }

    void register_free_host(size_t size, f64 timed) {

        mem_perf_infos.allocated_byte_host -= size;
        mem_perf_infos.time_free_host += timed;

        shambase::profiling::register_counter_val(
            "Host Memory", shambase::details::get_wtime(), mem_perf_infos.allocated_byte_host);
        shambase::profiling::register_counter_val(
            "Host free time", shambase::details::get_wtime(), mem_perf_infos.time_free_host);
    }

    template<sham::USMKindTarget target>
    std::string get_mode_name();

    template<>
    std::string get_mode_name<sham::device>() {
        return "device";
    }

    template<>
    std::string get_mode_name<sham::shared>() {
        return "shared";
    }

    template<>
    std::string get_mode_name<sham::host>() {
        return "host";
    }

} // namespace

namespace sham::details {

    MemPerfInfos get_mem_perf_info() { return mem_perf_infos; }

    void reset_mem_info_max() {
        mem_perf_infos.max_allocated_byte_host   = mem_perf_infos.allocated_byte_host;
        mem_perf_infos.max_allocated_byte_device = mem_perf_infos.allocated_byte_device;
        mem_perf_infos.max_allocated_byte_shared = mem_perf_infos.allocated_byte_shared;
    }

    std::string log_mem_perf_info(const std::shared_ptr<DeviceScheduler> &dev_sched) {

        std::string fmt = R"log(
    World infos :
        World size = {}
        World rank = {}
    Device infos :
        Device name = {}
    Allocs :
        max_allocated_byte_host = {}
        max_allocated_byte_device = {}
        max_allocated_byte_shared = {}
        allocated_byte_host = {}
        allocated_byte_device = {}
        allocated_byte_shared = {}
        )log";

        return shambase::format(
            fmt,
            shamcomm::world_size(),
            shamcomm::world_rank(),
            dev_sched->ctx->device->dev.get_info<sycl::info::device::name>(),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_host),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_device),
            shambase::readable_sizeof(mem_perf_infos.max_allocated_byte_shared),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_host),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_device),
            shambase::readable_sizeof(mem_perf_infos.allocated_byte_shared));
    }

    template<USMKindTarget target>
    void internal_free(void *usm_ptr, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched) {

        StackEntry __st{};

        f64 start_time = shambase::details::get_wtime();

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle",
            "free usm pointer size :",
            sz,
            " | ptr =",
            usm_ptr,
            " | mode =",
            get_mode_name<target>());

        sycl::context &sycl_ctx = dev_sched->ctx->ctx;
        sycl::free(usm_ptr, sycl_ctx);

        f64 end_time = shambase::details::get_wtime();

        if constexpr (target == device) {
            register_free_device(sz, end_time - start_time);
        } else if constexpr (target == shared) {
            register_free_shared(sz, end_time - start_time);
        } else if constexpr (target == host) {
            register_free_host(sz, end_time - start_time);
        }
    }

    template<USMKindTarget target>
    void *internal_alloc(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment) {

        StackEntry __st{};
        f64 start_time = shambase::details::get_wtime();

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle", "alloc usm pointer size :", sz, " | mode =", get_mode_name<target>());

        auto &ds                = shambase::get_check_ref(dev_sched);
        sycl::context &sycl_ctx = ds.ctx->ctx;
        sycl::device &dev       = ds.ctx->device->dev;

        void *usm_ptr = nullptr;

        auto catch_alloc_except = [&](auto alloc_lambda) {
            try {
                usm_ptr = alloc_lambda();
            } catch (std::exception &ex) {
                std::string log = shambase::format(
                    "Alloc failed with exception : {}\nShamrock mem infos : {}",
                    ex.what(),
                    log_mem_perf_info(dev_sched));
                shambase::throw_with_loc<std::runtime_error>(log);
            }
        };

        // check max alloc sizes
        if constexpr (target == device) {
            if (sz > ds.get_queue().get_device_prop().max_mem_alloc_size_dev) {
                std::string err_log = shambase::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "device\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    ds.get_queue().get_device_prop().max_mem_alloc_size_dev);
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else if constexpr (target == shared) {
            size_t max_alloc_size_dev  = ds.get_queue().get_device_prop().max_mem_alloc_size_dev;
            size_t max_alloc_size_host = ds.get_queue().get_device_prop().max_mem_alloc_size_host;
            if (sz > sycl::min(max_alloc_size_dev, max_alloc_size_host)) {
                std::string err_log = shambase::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "device\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    sycl::min(max_alloc_size_dev, max_alloc_size_host));
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else if constexpr (target == host) {
            if (sz > ds.get_queue().get_device_prop().max_mem_alloc_size_host) {
                std::string err_log = shambase::format(
                    "You are trying to allocate more than the maximum allocation size allowed by "
                    "the "
                    "host\n"
                    "  size = {} | max_alloc_size = {}",
                    sz,
                    ds.get_queue().get_device_prop().max_mem_alloc_size_host);
                shambase::throw_with_loc<std::runtime_error>(err_log);
            }
        } else {
            shambase::throw_unimplemented();
        }

        if (alignment) {

            if (*alignment % ds.get_queue().get_device_prop().mem_base_addr_align != 0) {
                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "The alignment of the USM pointer is not aligned with minimum device "
                    "alignment\n"
                    "  alignment = {} | device alignment = {} | alignment % device alignment = {}",
                    *alignment,
                    ds.get_queue().get_device_prop().mem_base_addr_align,
                    *alignment % ds.get_queue().get_device_prop().mem_base_addr_align));
            }

            if (sz % *alignment != 0) {
                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "The size of the USM pointer is not aligned with the given alignment\n"
                    "  size = {} | alignment = {} | size % alignment = {}",
                    sz,
                    *alignment,
                    sz % *alignment));
            }

            // TODO upgrade alignment to 256-bit for CUDA ?

            if constexpr (target == device) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_device(*alignment, sz, dev, sycl_ctx);
                });
            } else if constexpr (target == shared) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_shared(*alignment, sz, dev, sycl_ctx);
                });
            } else if constexpr (target == host) {
                catch_alloc_except([&] {
                    return sycl::aligned_alloc_host(*alignment, sz, sycl_ctx);
                });
            } else {
                shambase::throw_unimplemented();
            }
        } else {
            if constexpr (target == device) {
                catch_alloc_except([&] {
                    return sycl::malloc_device(sz, dev, sycl_ctx);
                });
            } else if constexpr (target == shared) {
                catch_alloc_except([&] {
                    return sycl::malloc_shared(sz, dev, sycl_ctx);
                });
            } else if constexpr (target == host) {
                catch_alloc_except([&] {
                    return sycl::malloc_host(sz, sycl_ctx);
                });
            } else {
                shambase::throw_unimplemented();
            }
        }

        if (usm_ptr == nullptr) {
            std::string err_msg = "";
            if (alignment) {
                err_msg = shambase::format(
                    "USM allocation failed, details : sz={}, target={}, alignment={}, alloc "
                    "result = {}",
                    sz,
                    get_mode_name<target>(),
                    *alignment,
                    usm_ptr);
            } else {
                err_msg = shambase::format(
                    "USM allocation failed, details : sz={}, target={}, alloc result = {}",
                    sz,
                    get_mode_name<target>(),
                    usm_ptr);
            }
            shambase::throw_with_loc<std::runtime_error>(err_msg + log_mem_perf_info(dev_sched));
        }

        if (alignment) {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle", "pointer created : ptr =", usm_ptr, "alignment =", *alignment);

            if (!shambase::is_aligned(usm_ptr, *alignment)) {
                shambase::throw_with_loc<std::runtime_error>(
                    "The pointer is not aligned with the given alignment");
            }

        } else {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle", "pointer created : ptr =", usm_ptr, "alignment = None");
        }

        f64 end_time = shambase::details::get_wtime();

        if constexpr (target == device) {
            register_alloc_device(sz, end_time - start_time);
        } else if constexpr (target == shared) {
            register_alloc_shared(sz, end_time - start_time);
        } else if constexpr (target == host) {
            register_alloc_host(sz, end_time - start_time);
        }

        return usm_ptr;
    }

#ifndef DOXYGEN
    template void internal_free<host>(
        void *usm_ptr, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);
    template void *internal_alloc<host>(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template void internal_free<device>(
        void *usm_ptr, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);
    template void *internal_alloc<device>(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template void internal_free<shared>(
        void *usm_ptr, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);
    template void *internal_alloc<shared>(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
#endif

} // namespace sham::details
