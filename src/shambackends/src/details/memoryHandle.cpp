// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file memoryHandle.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/USMPtrHolder.hpp"
#include "shamcomm/logs.hpp"
#include <shambackends/details/memoryHandle.hpp>

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

        shamcomm::logs::debug_alloc_ln(
            "memoryHandle",
            "create usm pointer size :",
            size,
            " | mode =",
            get_mode_name<target>());

        auto ret = USMPtrHolder<target>::create(size, dev_sched, alignment);

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
