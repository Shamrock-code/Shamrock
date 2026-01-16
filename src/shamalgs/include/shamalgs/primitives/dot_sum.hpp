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
 * @file dot_sum.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Provides functions to compute the sum of dot products of elements in a device buffer with
 * themselves.
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Compute the sum of dot products of elements in a device buffer with themselves.
     *
     * @tparam T The data type of elements in the buffer (e.g., float, double, int).
     * @param buf1 The input buffer containing the elements to sum.
     * @param start_id The starting index (inclusive) of the range to sum.
     * @param end_id The ending index (exclusive) of the range to sum.
     * @return shambase::VecComponent<T> The computed sum of dot products.
     *
     * Example:
     * @code{.cpp}
     * auto sched = shamsys::get_compute_Scheduler_ptr();
     * sham::DeviceBuffer<f64> values = ...;
     * f64 result = shamalgs::primitives::dot_sum(sched, values, 0, values.get_size());
     * @endcode
     */
    template<class T>
    shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1, u32 start_id, u32 end_id);

    // alias for dot_sum(buf1, 0, buf1.get_size())
    template<class T>
    inline shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1) {
        return dot_sum(buf1, 0, buf1.get_size());
    }

} // namespace shamalgs::primitives
