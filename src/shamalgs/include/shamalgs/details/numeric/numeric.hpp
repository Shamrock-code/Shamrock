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
 * @file numeric.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"

/**
 * @brief namespace containing the numeric algorithms of shamalgs
 *
 */
namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    sham::DeviceBuffer<T>
    exclusive_sum(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<T> &buf1, u32 len);

    template<class T>
    sycl::buffer<T> inclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len);

    template<class T>
    void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    template<class T>
    void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf, u32 len);

    /**
     * @brief Stream compaction algorithm
     *
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return std::tuple<sycl::buffer<u32>, u32> table of the index to extract and its size
     */
    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len);

    /**
     * @brief Stream compaction algorithm
     *
     * @param q the queue to run on
     * @param buf_flags buffer of only 0 and ones
     * @param len the length of the buffer considered
     * @return std::tuple<sycl::buffer<u32>, u32> table of the index to extract and its size
     */
    sham::DeviceBuffer<u32>
    stream_compact(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<u32> &buf_flags, u32 len);

} // namespace shamalgs::numeric
