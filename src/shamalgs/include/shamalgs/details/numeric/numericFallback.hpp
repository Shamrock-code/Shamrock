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
 * @file numericFallback.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::numeric::details {

    /**
     * @brief Exclusive sum fallback on SYCL buffer
     * @param q The queue to use for the fallback
     * @param buf1 The buffer to sum
     * @param len The length of the sum
     * @return A new buffer which is the output of the sum
     */
    template<class T>
    sycl::buffer<T> exclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    /**
     * @brief Exclusive sum fallback on USM
     * @param sched The scheduler for this fallback
     * @param buf1 The buffer to sum
     * @param len The length of the sum
     * @return A new buffer which is the output of the sum
     */
    template<class T>
    sham::DeviceBuffer<T> exclusive_sum_fallback_usm(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> inclusive_sum_fallback(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    void exclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    template<class T>
    void inclusive_sum_in_place_fallback(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    /**
     * @brief Stream compaction algorithm on fallback
     *
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return std::tuple<std::optional<sycl::buffer<u32>>, u32> table of the index to extract and
     * the length of it
     */
    std::tuple<std::optional<sycl::buffer<u32>>, u32> stream_compact_fallback(
        sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

    /**
     * @brief Stream compaction algorithm
     *
     * @param sched the scheduler to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return sham::DeviceBuffer<u32> table of the index to extract
     */
    sham::DeviceBuffer<u32> stream_compact_fallback(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric::details
