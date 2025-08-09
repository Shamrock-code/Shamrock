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
 * @file is_all_true.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Boolean reduction algorithm for checking if all elements are non-zero
 *
 * This header provides parallel algorithms to check if all elements in a buffer
 * are non-zero (logically true). The functions perform a boolean reduction operation
 * across the entire buffer, returning true only if every element satisfies the
 * non-zero condition.
 *
 * The algorithms come in two variants:
 * - `is_all_true` for sycl::buffer: Direct buffer-based processing
 * - `is_all_true` for sham::DeviceBuffer: USM-based processing
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Check if all elements in a sycl::buffer are non-zero
     *
     * Performs a boolean reduction operation to determine if every element in the
     * buffer is non-zero (logically true). The function returns true only if all
     * elements satisfy the condition `element != 0`.
     *
     * @tparam T Element type - must support comparison with zero
     * @param buf Buffer containing the elements to check
     * @param cnt Number of elements to check from the beginning of the buffer
     * @return true if all elements are non-zero, false otherwise
     *
     * @note Currently implemented on CPU but marked for GPU optimization
     * @note The function only checks the first `cnt` elements of the buffer
     *
     * @code
     * // Example: Check if all elements are non-zero
     * std::vector<i32> data = {1, 2, 3, 4, 5};
     * sycl::buffer<i32> buffer(data);
     *
     * // Check all elements
     * bool all_nonzero = is_all_true(buffer, data.size());
     * // Result: true (all elements are non-zero)
     *
     * // Example with zeros
     * std::vector<i32> data_with_zero = {1, 0, 3, 4};
     * sycl::buffer<i32> buffer2(data_with_zero);
     * bool has_zero = is_all_true(buffer2, data_with_zero.size());
     * // Result: false (contains zero)
     * @endcode
     */
    template<class T>
    bool is_all_true(sycl::buffer<T> &buf, u32 cnt);

    /**
     * @brief Check if all elements in a sham::DeviceBuffer are non-zero
     *
     * Performs a boolean reduction operation to determine if every element in the
     * device buffer is non-zero (logically true). The function returns true only
     * if all elements satisfy the condition `element != 0`.
     *
     * @tparam T Element type - must support comparison with zero
     * @param buf Device buffer containing the elements to check
     * @param cnt Number of elements to check from the beginning of the buffer
     * @return true if all elements are non-zero, false otherwise
     *
     * @note Currently implemented by copying to sycl::buffer and using CPU processing
     * @note The function only checks the first `cnt` elements of the buffer
     *
     * @code
     * // Example: Check device buffer elements
     * std::vector<u8> flags = {1, 1, 1, 1};
     * sham::DeviceBuffer<u8> device_buffer(flags);
     *
     * // Check all flags
     * bool all_set = is_all_true(device_buffer, flags.size());
     * // Result: true (all flags are set)
     *
     * // Example with boolean conditions
     * std::vector<i32> conditions = {5 > 0, 3 > 0, 0 > 0, 1 > 0};
     * sham::DeviceBuffer<i32> cond_buffer(conditions);
     * bool all_conditions_met = is_all_true(cond_buffer, conditions.size());
     * // Result: false (third condition evaluates to 0)
     * @endcode
     */
    template<class T>
    bool is_all_true(sham::DeviceBuffer<T> &buf, u32 cnt);

} // namespace shamalgs::primitives
