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
 * @file digit_histogram.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Histograms of every radix digit place of a key buffer, in a single read of the keys.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Compute the histograms of all the digit places of the first `len` keys
     *
     * The keys are split in `npasses = bitsizeof(Tkey) / radix_bits` digits of `radix_bits`
     * bits, digit place `p` being `(key >> (p * radix_bits)) & (2^radix_bits - 1)` (place 0 is
     * the least significant digit). All the digit places are counted in a single read of the
     * keys (this is the upfront histogram of the Onesweep radix sort).
     *
     * `buf_hist` is resized to `npasses * nbuckets` (`nbuckets = 2^radix_bits`) and holds the
     * histograms digit place major :
     *
     * \code{.cpp}
     * buf_hist[p * nbuckets + digit] = number of keys k with ((k >> (p * radix_bits)) &
     *                                  (nbuckets - 1)) == digit
     * \endcode
     *
     * or equivalently, starting from `buf_hist` filled with zeros :
     *
     * \code{.cpp}
     * for (u32 i = 0; i < len; i++) {
     *     for (u32 digit_place = 0; digit_place < npasses; digit_place++) {
     *         u32 digit_val = (buf_key[i] >> (digit_place * radix_bits)) & (nbuckets - 1);
     *         buf_hist[digit_val + digit_place * nbuckets] += 1;
     *     }
     * }
     * \endcode
     *
     * so every digit place histogram is contiguous and sums to `len`.
     *
     * Example with `Tkey = u32`, `radix_bits = 8` : 4 histograms of 256 bins,
     * `buf_hist[0 .. 255]` for bits 0-7, ..., `buf_hist[768 .. 1023]` for bits 24-31.
     *
     * @tparam Tkey unsigned integer key type
     * @tparam radix_bits number of bits of a digit, must divide the bit size of `Tkey` (at most
     * 8 so that the local histograms fit in local memory)
     * @param sched the device scheduler to run on
     * @param buf_key the keys
     * @param buf_hist the resulting histograms (resized by the function)
     * @param len number of keys to count, from the beginning of `buf_key`
     */
    template<class Tkey, u32 radix_bits>
    void digit_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<u32> &buf_hist,
        u32 len);

    /**
     * @brief Compute the histograms of all the digit places of every key of `buf_key`
     *
     * Same as the overload taking `len`, with `len = buf_key.get_size()`. See it for the layout
     * of `buf_hist`.
     */
    template<class Tkey, u32 radix_bits>
    void digit_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<u32> &buf_hist);

} // namespace shamalgs::primitives
