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
 * @file sort_by_keys.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sort by keys algorithms
 *
 * This header provides parallel sorting algorithms that sort key-value pairs
 * based on the key values. The algorithms are optimized for GPU execution
 * using sycl::buffers or USM.
 *
 * The sorting functions come in two variants:
 * - `sort_by_key_pow2_len`: Optimized for power-of-2 buffer lengths
 * - `sort_by_key`: General case
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Sort key-value pairs using sycl::buffers (power-of-2 optimized)
     *
     * Performs an in-place parallel sort of key-value pairs where the values
     * are reordered according to the sorted order of their corresponding keys.
     *
     * @tparam Tkey Key type - must be comparable (supports < operator)
     * @tparam Tval Value type - can be any copyable type
     * @param q sycl::queue for device execution
     * @param buf_key Buffer containing the keys to sort by
     * @param buf_values Buffer containing the values to reorder
     * @param len Length of both buffers (must be a power of 2)
     *
     * @note The function modifies both buffers in-place
     *
     * @code
     * // Example: Sort data by keys
     * sycl::queue q;
     * sycl::buffer<float> keys(input_keys, N);
     * sycl::buffer<DataType> values(input_values, N);
     *
     * // Sort values according to key order
     * sort_by_key_pow2_len(q, keys, values, N);
     * @endcode
     */
    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len);

    /**
     * @brief Sort key-value pairs using USM buffers (power-of-2 optimized)
     *
     * Performs an in-place parallel sort of key-value pairs where the values
     * are reordered according to the sorted order of their corresponding keys.
     *
     * @tparam Tkey Key type - must be comparable (supports < operator)
     * @tparam Tval Value type - can be any copyable type
     * @param sched sham::DeviceScheduler_ptr for execution
     * @param buf_key Device buffer containing the keys to sort by
     * @param buf_values Device buffer containing the values to reorder
     * @param len Length of both buffers (must be a power of 2)
     *
     * @note The function modifies both buffers in-place
     *
     * @code
     * // Example: Sort data by keys using USM buffers
     * auto sched = shamsys::instance::get_compute_scheduler_ptr();
     * sham::DeviceBuffer<float> keys(input_keys, N);
     * sham::DeviceBuffer<DataType> values(input_values, N);
     *
     * // Sort values according to key order
     * sort_by_key_pow2_len(sched, keys, values, N);
     * @endcode
     */
    template<class Tkey, class Tval>
    void sort_by_key_pow2_len(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len);

    /**
     * @brief Sort key-value pairs using sycl::buffers
     *
     * Performs an in-place parallel sort of key-value pairs where the values
     * are reordered according to the sorted order of their corresponding keys.
     *
     * @tparam Tkey Key type - must be comparable (supports < operator)
     * @tparam Tval Value type - can be any copyable type
     * @param q sycl::queue for device execution
     * @param buf_key Buffer containing the keys to sort by
     * @param buf_values Buffer containing the values to reorder
     * @param len Length of both buffers
     *
     * @throws std::invalid_argument if len is not a power of 2
     *
     * @note The function modifies both buffers in-place
     * @note This function currently only supports powers of 2
     *
     * @code
     * // Example: Sort data by keys
     * sycl::queue q;
     * sycl::buffer<double> keys(input_keys, N);
     * sycl::buffer<DataType> values(input_values, N);
     *
     * // Sort values according to key order
     * sort_by_key(q, keys, values, N);
     * @endcode
     */
    template<class Tkey, class Tval>
    void sort_by_key(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {
        if (!shambase::is_pow_of_two(len))
            shambase::throw_with_loc<std::invalid_argument>("Length must be a power of 2");
        sort_by_key_pow2_len(q, buf_key, buf_values, len);
    }

    /**
     * @brief Sort key-value pairs using USM buffers
     *
     * Performs an in-place parallel sort of key-value pairs where the values
     * are reordered according to the sorted order of their corresponding keys.
     *
     * @tparam Tkey Key type - must be comparable (supports < operator)
     * @tparam Tval Value type - can be any copyable type
     * @param sched sham::DeviceScheduler_ptr for execution
     * @param buf_key Device buffer containing the keys to sort by
     * @param buf_values Device buffer containing the values to reorder
     * @param len Length of both buffers
     *
     * @throws std::invalid_argument if len is not a power of 2
     *
     * @note The function modifies both buffers in-place
     * @note This function currently only supports powers of 2
     *
     * @code
     * // Example: Sort data by keys using USM buffers
     * auto sched = shamsys::instance::get_compute_scheduler_ptr();
     * sham::DeviceBuffer<double> keys(input_keys, N);
     * sham::DeviceBuffer<DataType> values(input_values, N);
     *
     * // Sort values according to key order
     * sort_by_key(sched, keys, values, N);
     * @endcode
     */
    template<class Tkey, class Tval>
    void sort_by_key(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {
        if (!shambase::is_pow_of_two(len))
            shambase::throw_with_loc<std::invalid_argument>("Length must be a power of 2");
        sort_by_key_pow2_len(sched, buf_key, buf_values, len);
    }

} // namespace shamalgs::primitives
