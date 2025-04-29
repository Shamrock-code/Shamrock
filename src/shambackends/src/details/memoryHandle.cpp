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

#include "shambase/aliases_float.hpp"
#include "shambase/profiling/chrome.hpp"
#include "shambase/profiling/profiling.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/MemPerfInfos.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include <shambackends/details/memoryHandle.hpp>

namespace sham::details {

    template<USMKindTarget target>
    USMPtrHolder<target> create_usm_ptr(
        size_t size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment) {

        StackEntry __st{};

        auto create = [&]() {
            if (size > 0) {
                return USMPtrHolder<target>::create(size, dev_sched, alignment);
            } else {
                return USMPtrHolder<target>::create_nullptr(dev_sched);
            }
        };

        return create();
    }

    template<USMKindTarget target>
    void
    release_usm_ptr(USMPtrHolder<target> &&usm_ptr_hold, details::BufferEventHandler &&events) {

        StackEntry __st{};

        shamcomm::logs::debug_alloc_ln("memoryHandle", "waiting event completion ...");
        events.wait_all();
        shamcomm::logs::debug_alloc_ln("memoryHandle", "done, freeing memory");
        usm_ptr_hold.free_ptr();
    }

#ifndef DOXYGEN
    template USMPtrHolder<device> create_usm_ptr<device>(
        size_t size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template USMPtrHolder<shared> create_usm_ptr<shared>(
        size_t size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);
    template USMPtrHolder<host> create_usm_ptr<host>(
        size_t size, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);

    template void release_usm_ptr<device>(
        USMPtrHolder<device> &&usm_ptr_hold, details::BufferEventHandler &&events);
    template void release_usm_ptr<shared>(
        USMPtrHolder<shared> &&usm_ptr_hold, details::BufferEventHandler &&events);
    template void
    release_usm_ptr<host>(USMPtrHolder<host> &&usm_ptr_hold, details::BufferEventHandler &&events);

#endif

} // namespace sham::details
