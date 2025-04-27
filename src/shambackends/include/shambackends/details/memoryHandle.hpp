// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file memoryHandle.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief This file contains the declaration of the memory handling and its methods.
 *
 * The memoryHandle class is used to allocate SYCL memory. It provides a way to safely allocate,
 * use, and deallocate memory in SYCL. This file should always be used to allocate SYCL memory, as
 * it ensures correct memory management (including memory pooling).
 *
 * @section Usage
 *
 * To allocate SYCL memory, use the `create_usm_ptr` method . Here is an example:
 *
 * @code{.cpp}
 * // Allocate 1024 bytes of SYCL device memory
 * auto ptr = sham::details::create_usm_ptr<device>(1024, device_scheduler);
 * @endcode
 *
 * After using the SYCL memory, make sure to deallocate it using the `deallocate_usm_ptr` method.
 *
 * @code{.cpp}
 * // Give memory back to the memory manager (potentially pooling memory)
 * sham::details::memoryHandle::deallocate_usm_ptr(ptr_holder, events);
 * @endcode
 *
 * @section Implementation details
 *
 * The memoryHandle class uses the USMPtrHolder class to manage the memory. The USMPtrHolder class
 * is a smart pointer that manages the memory allocated using SYCL unified shared memory (USM). It
 * provides a way to safely allocate, use, and deallocate memory in USM.
 */

#include "shambackends/MemPerfInfos.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/details/BufferEventHandler.hpp"
#include "shambackends/details/internal_alloc.hpp"

namespace sham::details {

    /**
     * @brief Create a USM pointer with at least the given size in bytes.
     *
     * @note The USM pointer may have a larger allocation than the required size.
     *
     * @todo should be renamed to aquire_...
     *
     * @tparam target The target of the USM pointer.
     * @param size The size of the pointer in bytes.
     * @param dev_sched Pointer to the device scheduler.
     * @param alignment The alignment of the USM pointer (optional).
     *
     * @return USMPtrHolder<target> The newly created USM pointer.
     */
    template<USMKindTarget target>
    USMPtrHolder<target> create_usm_ptr(
        size_t size,
        std::shared_ptr<DeviceScheduler> dev_sched,
        std::optional<size_t> alignment = std::nullopt);

    /**
     * @brief Release a USM pointer.
     *
     * @tparam target The target of the USM pointer.
     * @param usm_ptr_hold The USM pointer holder to release.
     * @param events The event handler to wait for completion.
     */
    template<USMKindTarget target>
    void release_usm_ptr(USMPtrHolder<target> &&usm_ptr_hold, details::BufferEventHandler &&events);

} // namespace sham::details
