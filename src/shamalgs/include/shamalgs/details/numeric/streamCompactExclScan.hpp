// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file streamCompactExclScan.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    /**
     * @brief Stream compaction algorithm using exclusive summation
     *
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return sycl::buffer<u32> table of the index to extract
     */
    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact_excl_scan(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

    /**
     * @brief Performs stream compaction using exclusive scan on a device.
     *
     * @param sched Device scheduler that manages the computation queue.
     * @param buf_flags Buffer containing flags (0s and 1s) indicating elements to keep.
     * @param len Length of the buffer, representing the number of elements to process.
     * @return A DeviceBuffer containing the indices of the elements to keep.
     */
    sham::DeviceBuffer<u32> stream_compact_excl_scan(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric::details
