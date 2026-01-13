// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file internal_alloc.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the methods to actually allocate memory
 */

#include "shambackends/MemPerfInfos.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/details/BufferEventHandler.hpp"

namespace sham::details {

    /**
     * @brief Allocate a USM pointer with at least the given size in bytes.
     *
     * @param sz The minimum size of the USM pointer in bytes.
     * @param dev_sched The SYCL queue used to allocate the USM pointer.
     * @param alignment The alignment of the USM pointer (optional).
     *
     * @returns A pointer to the allocated USM memory.
     */
    template<USMKindTarget target>
    void *internal_alloc(
        size_t sz, std::shared_ptr<DeviceScheduler> dev_sched, std::optional<size_t> alignment);

    /**
     * @brief Free a USM pointer.
     *
     * @param usm_ptr The pointer to free.
     * @param sz The size of the USM pointer in bytes.
     * @param dev_sched The SYCL queue used to free the USM pointer.
     */
    template<USMKindTarget target>
    void internal_free(void *usm_ptr, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);

    /// @brief Retrieve the memory performance information.
    /// @return A MemPerfInfos object containing the memory performance data.
    MemPerfInfos get_mem_perf_info();

    /// @brief Reset the memory information for the maximum allocated bytes.
    void reset_mem_info_max();

} // namespace sham::details
