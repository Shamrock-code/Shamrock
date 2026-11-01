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
 * This header provides a parallel sorting algorithm that sorts key-value
 * pairs based on the key values.
 *
 * `sort_by_keys` supports any buffer length and selects its implementation
 * through the generic implementation selector mechanism (see ImplVariant.hpp).
 */

#include "shambackends/DeviceBuffer.hpp"

namespace shamalgs::primitives {

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

} // namespace shamalgs::primitives
