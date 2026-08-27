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
 * `sort_by_key_pow2_len` only supports buffer lengths that are a power of 2
 * (callers must round up beforehand, e.g. with `shambase::roundup_pow2`).
 *
 * `sort_by_keys` supports any buffer length and selects its implementation
 * through the generic implementation selector mechanism (see ImplVariant.hpp).
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include <string>
#include <vector>

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
     * @brief Sort key-value pairs using USM buffers (general length)
     *
     * Performs an in-place sort of key-value pairs where the values are
     * reordered according to the sorted order of their corresponding keys.
     * Unlike `sort_by_key_pow2_len`, `len` does not need to be a power of 2.
     *
     * The implementation used is selected through the `impl` sub-namespace
     * below (see ImplVariant.hpp for the generic mechanism).
     *
     * @tparam Tkey Key type - must be comparable (supports < operator)
     * @tparam Tval Value type - can be any copyable type
     * @param buf_key Device buffer containing the keys to sort by
     * @param buf_values Device buffer containing the values to reorder
     * @param len Length of both buffers
     *
     * @note The function modifies both buffers in-place
     */
    template<class Tkey, class Tval>
    void sort_by_keys(
        sham::DeviceBuffer<Tkey> &buf_key, sham::DeviceBuffer<Tval> &buf_values, u32 len);

    /// namespace to control implementation behavior
    namespace impl {

        /// Get list of available sort by keys implementations, as config json strings
        std::vector<std::string> get_default_impl_list_sort_by_keys();

        /// Get the current implementation for sort by keys, as a config json string
        std::string get_current_impl_sort_by_keys();

        /// Check if an implementation has been selected for sort by keys
        bool is_impl_set_sort_by_keys();

        /// Set the implementation for sort by keys, from a config json string
        void set_impl_sort_by_keys(const std::string &impl);

        /// Select the default implementation for sort by keys
        void autoselect_impl_sort_by_keys();

    } // namespace impl

} // namespace shamalgs::primitives
