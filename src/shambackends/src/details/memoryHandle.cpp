// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file memoryHandle.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/profiling/chrome.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <shambackends/details/memoryHandle.hpp>

namespace {
    size_t allocated_byte_device = 0;
    size_t allocated_byte_shared = 0;
    size_t allocated_byte_host   = 0;

    void register_alloc_device(size_t size) {
        allocated_byte_device += size;
        shambase::profiling::register_counter_val(
            "Device Memory", shambase::details::get_wtime(), allocated_byte_device);
    }

    void register_alloc_shared(size_t size) {
        allocated_byte_shared += size;
        shambase::profiling::register_counter_val(

            "Shared Memory", shambase::details::get_wtime(), allocated_byte_shared);
    }

    void register_alloc_host(size_t size) {
        allocated_byte_host += size;
        shambase::profiling::register_counter_val(
            "Host Memory", shambase::details::get_wtime(), allocated_byte_host);
    }

    void register_free_device(size_t size) {
        allocated_byte_device -= size;
        shambase::profiling::register_counter_val(

            "Device Memory", shambase::details::get_wtime(), allocated_byte_device);
    }

    void register_free_shared(size_t size) {
        allocated_byte_shared -= size;
        shambase::profiling::register_counter_val(

            "Shared Memory", shambase::details::get_wtime(), allocated_byte_shared);
    }

    void register_free_host(size_t size) {
        allocated_byte_host -= size;
        shambase::profiling::register_counter_val(

            "Host Memory", shambase::details::get_wtime(), allocated_byte_host);
    }

} // namespace

namespace sham::details {

    template<USMKindTarget target>
    std::string get_mode_name();

    template<>
    std::string get_mode_name<device>() {
        return "device";
    }

    template<>
    std::string get_mode_name<shared>() {
        return "shared";
    }

    template<>
    std::string get_mode_name<host>() {
        return "host";
    }

    template<USMKindTarget target>
    USMPtrHolder<target> create_usm_ptr(
        u32 size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment) {

        StackEntry __st{};

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle",
            "create usm pointer size :",
            size,
            " | mode =",
            get_mode_name<target>());

        auto ret = USMPtrHolder<target>::create(size, dev_sched, alignment);

        if constexpr (target == device) {
            register_alloc_device(size);
        } else if constexpr (target == shared) {
            register_alloc_shared(size);
        } else if constexpr (target == host) {
            register_alloc_host(size);
        }

        if (alignment) {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle",
                "pointer created : ptr =",
                ret.get_raw_ptr(),
                "alignment =",
                *alignment);

            if (!shambase::is_aligned(ret.get_raw_ptr(), *alignment)) {
                shambase::throw_with_loc<std::runtime_error>(
                    "The pointer is not aligned with the given alignment");
            }

        } else {

            shamcomm::logs::debug_alloc_ln(
                "memoryHandle", "pointer created : ptr =", ret.get_raw_ptr(), "alignment = None");
        }

        return ret;
    }

    template<USMKindTarget target>
    void
    release_usm_ptr(USMPtrHolder<target> &&usm_ptr_hold, details::BufferEventHandler &&events) {

        StackEntry __st{};

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle",
            "release usm pointer size :",
            usm_ptr_hold.get_bytesize(),
            " | ptr =",
            usm_ptr_hold.get_raw_ptr(),
            " | mode =",
            get_mode_name<target>());
        shamcomm::logs::debug_alloc_ln("memoryHandle", "waiting event completion ...");
        events.wait_all();
        shamcomm::logs::debug_alloc_ln("memoryHandle", "done, freeing memory");
        usm_ptr_hold.free_ptr();

        if constexpr (target == device) {
            register_free_device(usm_ptr_hold.get_bytesize());
        } else if constexpr (target == shared) {
            register_free_shared(usm_ptr_hold.get_bytesize());
        } else if constexpr (target == host) {
            register_free_host(usm_ptr_hold.get_bytesize());
        }
    }

    template USMPtrHolder<device> create_usm_ptr<device>(
        u32 size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template USMPtrHolder<shared> create_usm_ptr<shared>(
        u32 size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template USMPtrHolder<host> create_usm_ptr<host>(
        u32 size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);

    template void release_usm_ptr<device>(
        USMPtrHolder<device> &&usm_ptr_hold, details::BufferEventHandler &&events);
    template void release_usm_ptr<shared>(
        USMPtrHolder<shared> &&usm_ptr_hold, details::BufferEventHandler &&events);
    template void
    release_usm_ptr<host>(USMPtrHolder<host> &&usm_ptr_hold, details::BufferEventHandler &&events);

} // namespace sham::details
