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
 * @file batcherOddEvenSort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Batcher odd-even mergesort, native for any length
 *
 * The bitonic sorting networks of `bitonicSort*.hpp` only exist for lengths that are powers
 * of two, which forces every caller to round its arrays up to `roundup_pow2(len)` and to pad
 * the tail with a sentinel greater than every key.
 *
 * Batcher's odd-even mergesort has no such restriction: all of its comparators are ascending
 * (there is no bitonic direction flag), so truncating the network to the comparators that
 * stay inside `[0, len)` is exactly equivalent to running the power of two network on an
 * array padded with `+infinity`, every dropped comparator being a no-op on such an array.
 * The implementations here therefore sort any length in place, with no padding and no
 * scratch allocation.
 *
 * The network is
 *
 *     for p = 1, 2, 4, ... while p < n
 *       for k = p, p/2, ... while k >= 1
 *         for j = k mod p to n-1-k step 2k
 *           for i = 0 to min(k-1, n-j-k-1)
 *             if floor((i+j)/2p) == floor((i+j+k)/2p)
 *               compare_exchange(a[i+j], a[i+j+k])
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <vector>

/**
 * @brief namespace to store algorithms implemented by shamalgs
 *
 */
namespace shamalgs::algorithm::details {

    /**
     * @brief Sort key-value pairs of any length using a Batcher odd-even merge network
     *
     * Both buffers are modified in place: the keys are sorted in ascending order and the
     * values follow the same permutation. Unlike the bitonic implementations, `len` is
     * unconstrained, and no padding buffer is allocated.
     *
     * @tparam Tkey Key type, must be comparable (supports operator<)
     * @tparam Tval Value type, must be copyable
     * @param sched Device scheduler used for the kernel launches
     * @param buf_key Device buffer holding the keys to sort by
     * @param buf_values Device buffer holding the values to reorder
     * @param len Length of both buffers, any value including 0 and 1
     *
     * @note The sort is not stable, equal keys may be reordered
     * @note One kernel is launched per stage of the network,
     * `ceil(log2(len))*(ceil(log2(len))+1)/2` in total, each with `ceil(len/2)` threads
     */
    template<class Tkey, class Tval>
    void sort_by_key_batcher_odd_even(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len);

    /**
     * @brief Host reference of @ref sort_by_key_batcher_odd_even
     *
     * A literal transcription of the network quoted at the top of this file, written as four
     * plain loops with no index algebra. It is the oracle the device implementation is tested
     * against: both run the very same comparators in the very same order, so their outputs
     * must match element for element even though the sort is unstable.
     *
     * @tparam Tkey Key type, must be comparable (supports operator<)
     * @tparam Tval Value type, must be swappable
     * @param keys Keys to sort by, sorted in place
     * @param values Values to reorder, permuted in place alongside the keys
     */
    template<class Tkey, class Tval>
    void sort_by_key_batcher_odd_even_host_reference(
        std::vector<Tkey> &keys, std::vector<Tval> &values);

} // namespace shamalgs::algorithm::details
