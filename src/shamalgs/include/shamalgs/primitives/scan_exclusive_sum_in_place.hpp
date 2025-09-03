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
 * @file scan_exclusive_sum_in_place.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief In-place exclusive scan (prefix sum) algorithm for device buffers
 *
 * This header provides parallel algorithms to compute the exclusive prefix sum
 * of elements in a device buffer. The exclusive scan operation computes for each
 * position i the sum of all elements from index 0 to i-1, with the first element
 * receiving the identity value (0 for addition).
 *
 * The algorithm performs the operation in-place, modifying the input buffer
 * directly without allocating additional memory for the result. This makes it
 * memory-efficient for large datasets.
 */

#include "shambase/aliases_int.hpp"
#include "shamalgs/impl_utils.hpp"
#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Compute exclusive prefix sum in-place on a device buffer
     *
     * Performs an exclusive scan (prefix sum) operation on the device buffer,
     * modifying it in-place. For each position i, the result contains the sum
     * of all elements from index 0 to i-1. The first element is set to 0 (the
     * identity for addition).
     *
     * The operation transforms: [a, b, c, d] → [0, a, a+b, a+b+c]
     *
     * @tparam T Element type - must support addition and assignment
     * @param buf1 Device buffer to scan in-place (modified by this operation)
     * @param len Number of elements to process from the beginning of the buffer
     *
     * @note The buffer is modified in-place; original values are overwritten
     *
     * @code{.cpp}
     * // Example: Basic exclusive scan
     * std::vector<u32> data = {1, 2, 3, 4, 5};
     * sham::DeviceBuffer<u32> buffer(data);
     *
     * // Perform in-place exclusive scan
     * scan_exclusive_sum_in_place(buffer, data.size());
     *
     * // Buffer now contains: [0, 1, 3, 6, 10]
     * auto result = buffer.copy_to_stdvec();
     * // result[0] = 0, result[1] = 1, result[2] = 3, result[3] = 6, result[4] = 10
     *
     * // Example: Partial scan
     * std::vector<u32> values = {5, 10, 15, 20, 25, 30};
     * sham::DeviceBuffer<u32> partial_buffer(values);
     *
     * // Scan only first 4 elements
     * scan_exclusive_sum_in_place(partial_buffer, 4);
     *
     * // Buffer now contains: [0, 5, 15, 30, 25, 30]
     * // Only first 4 elements were modified
     * @endcode
     */
    template<class T>
    void scan_exclusive_sum_in_place(sham::DeviceBuffer<T> &buf1, u32 len);

    /// namespace to control implementation behavior
    namespace impl {

        /// Get list of available scan_exclusive_sum_in_place implementations
        std::vector<shamalgs::impl_param> get_default_impl_list_scan_exclusive_sum_in_place();

        /// Get the current implementation for scan_exclusive_sum_in_place
        shamalgs::impl_param get_current_impl_scan_exclusive_sum_in_place();

        /// Set the implementation for scan_exclusive_sum_in_place
        void set_impl_scan_exclusive_sum_in_place(
            const std::string &impl, const std::string &param = "");

    } // namespace impl

} // namespace shamalgs::primitives
