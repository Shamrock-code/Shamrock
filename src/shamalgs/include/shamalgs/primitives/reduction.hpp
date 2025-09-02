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
 * @file reduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Compute the sum of elements in a device buffer within a specified range.
     *
     * This function computes the sum of all elements in the buffer between start_id (inclusive)
     * and end_id (exclusive). The computation is performed on the device using the provided
     * scheduler.
     *
     * @tparam T The data type of elements in the buffer (e.g., float, double, int).
     * @param sched The device scheduler to run on.
     * @param buf1 The input buffer containing the elements to sum.
     * @param start_id The starting index (inclusive) of the range to sum.
     * @param end_id The ending index (exclusive) of the range to sum.
     * @return T The computed sum of elements in the specified range.
     *
     * @pre start_id <= end_id
     * @pre end_id <= buf1.get_size()
     *
     * Example:
     * @code{.cpp}
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     *
     * sham::DeviceBuffer<double> values = ...;
     * u32 start = 0;
     * u32 end = values.get_size();
     *
     * double total = shamalgs::primitives::sum(sched, values, start, end);
     * @endcode
     *
     * values = {1.0, 2.0, 3.0, 4.0, 5.0}, start = 1, end = 4
     * result = 9.0 (2.0 + 3.0 + 4.0)
     */
    template<class T>
    T sum(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    /**
     * @brief Find the minimum element in a device buffer within a specified range.
     *
     * This function finds the minimum value among all elements in the buffer between start_id
     * (inclusive) and end_id (exclusive). The computation is performed on the device using the
     * provided scheduler.
     *
     * @tparam T The data type of elements in the buffer (e.g., float, double, int).
     * @param sched The device scheduler to run on.
     * @param buf1 The input buffer containing the elements to search.
     * @param start_id The starting index (inclusive) of the range to search.
     * @param end_id The ending index (exclusive) of the range to search.
     * @return T The minimum value found in the specified range.
     *
     * @pre start_id < end_id (range must be non-empty)
     * @pre end_id <= buf1.get_size()
     *
     * Example:
     * @code{.cpp}
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     *
     * sham::DeviceBuffer<double> values = ...;
     * u32 start = 0;
     * u32 end = values.get_size();
     *
     * double minimum = shamalgs::primitives::min(sched, values, start, end);
     * @endcode
     *
     * values = {5.0, 2.0, 8.0, 1.0, 6.0}, start = 1, end = 4
     * result = 1.0 (minimum of {2.0, 8.0, 1.0})
     */
    template<class T>
    T min(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    /**
     * @brief Find the maximum element in a device buffer within a specified range.
     *
     * This function finds the maximum value among all elements in the buffer between start_id
     * (inclusive) and end_id (exclusive). The computation is performed on the device using the
     * provided scheduler.
     *
     * @tparam T The data type of elements in the buffer (e.g., float, double, int).
     * @param sched The device scheduler to run on.
     * @param buf1 The input buffer containing the elements to search.
     * @param start_id The starting index (inclusive) of the range to search.
     * @param end_id The ending index (exclusive) of the range to search.
     * @return T The maximum value found in the specified range.
     *
     * @pre start_id < end_id (range must be non-empty)
     * @pre end_id <= buf1.get_size()
     *
     * Example:
     * @code{.cpp}
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     *
     * sham::DeviceBuffer<double> values = ...;
     * u32 start = 0;
     * u32 end = values.get_size();
     *
     * double maximum = shamalgs::primitives::max(sched, values, start, end);
     * @endcode
     *
     * values = {5.0, 2.0, 8.0, 1.0, 6.0}, start = 1, end = 4
     * result = 8.0 (maximum of {2.0, 8.0, 1.0})
     */
    template<class T>
    T max(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id);

    namespace impl {

        /**
         * @brief Set the implementation for reduction algorithms
         *
         * Configures the specific implementation to use for reduction operations
         * (sum, min, max) across the shamalgs primitives. This allows fine-tuning
         * the performance characteristics of reduction algorithms based on the
         * target hardware or specific requirements.
         *
         * @param impl The name of the implementation to use
         * @param param Optional parameter string for implementation-specific configuration
         *
         * @note The available implementations can be queried using get_impl_list_reduction()
         * @note Invalid implementation names will be ignored
         *
         * @code{.cpp}
         * // Example: Set a specific reduction implementation
         * shamalgs::primitives::impl::set_impl_reduction("optimized_gpu");
         *
         * // Example: Set implementation with parameters
         * shamalgs::primitives::impl::set_impl_reduction("tiled", "tile_size=256");
         * @endcode
         */
        void set_impl_reduction(const std::string &impl, const std::string &param = "");

        /**
         * @brief Reset reduction implementation to default
         *
         * Restores the default implementation for reduction algorithms. This is
         * useful when custom implementations need to be reverted to the system
         * default behavior.
         *
         * @code{.cpp}
         * // Example: Reset to default implementation
         * shamalgs::primitives::impl::set_impl_reduction_default();
         * @endcode
         */
        void set_impl_reduction_default();

        /**
         * @brief Get list of available reduction implementations
         *
         * Returns a vector of strings containing the names of all available
         * reduction algorithm implementations that can be used with
         * set_impl_reduction().
         *
         * @return std::vector<std::string> List of available implementation names
         *
         * @code{.cpp}
         * // Example: Query available implementations
         * auto impls = shamalgs::primitives::impl::get_impl_list_reduction();
         * for (const auto& impl : impls) {
         *     std::cout << "Available: " << impl << std::endl;
         * }
         * @endcode
         */
        std::vector<std::string> get_impl_list_reduction();
    } // namespace impl

} // namespace shamalgs::primitives
