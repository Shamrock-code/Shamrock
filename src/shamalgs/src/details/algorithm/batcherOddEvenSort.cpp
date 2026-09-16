// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file batcherOddEvenSort.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Batcher odd-even mergesort, native for any length
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/details/algorithm/batcherOddEvenSort.hpp"
#include "shambackends/kernel_call.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace shamalgs::algorithm::details {

    /**
     * @brief Device side primitives of the odd-even merge network
     *
     * Held as static members so that they can be called from within a device lambda.
     */
    template<class Tkey, class Tval>
    struct OddEvenOrderingPrimitive {

        /// Ascending branchless compare-exchange of the pair (a, b), with a < b
        inline static void compare_exchange(
            Tkey *__restrict__ keys, Tval *__restrict__ vals, u32 a, u32 b) {

            Tkey key_a = keys[a];
            Tkey key_b = keys[b];
            Tval val_a = vals[a];
            Tval val_b = vals[b];

            bool swap = key_b < key_a;

            keys[a] = (swap) ? key_b : key_a;
            keys[b] = (swap) ? key_a : key_b;
            vals[a] = (swap) ? val_b : val_a;
            vals[b] = (swap) ? val_a : val_b;
        }

        /**
         * @brief Work of a single thread within the (p, k) stage of the network
         *
         * `p` and `k` are powers of two with `k` dividing `p`, so the two innermost loops of
         * the reference network collapse to the closed form below, made of shifts and masks
         * only. Thread `t` owns the comparator whose low index is
         *
         *     x = (k mod p) + 2*k*(t / k) + (t mod k)
         *
         * The two early returns are the guards of the reference network: `x + k >= len`
         * drops the comparators that would have touched the `+infinity` padding of the power
         * of two network, and the second one is the odd-even merge condition
         * `floor(x/2p) == floor((x+k)/2p)`.
         *
         * @param keys Keys to sort by
         * @param vals Values to reorder
         * @param len Length of both arrays
         * @param k Comparator distance of this stage
         * @param j0 Offset of the first comparator of this stage, `k mod p`
         * @param log_k Base two logarithm of `k`
         * @param log_2p Base two logarithm of `2*p`
         * @param t Index of the thread
         */
        inline static void merge_step(
            Tkey *__restrict__ keys,
            Tval *__restrict__ vals,
            u64 len,
            u64 k,
            u64 j0,
            u32 log_k,
            u32 log_2p,
            u64 t) {

            u64 x = j0 + ((t >> log_k) << (log_k + 1)) + (t & (k - 1));

            if (x + k >= len) {
                return; // comparator truncated away by the end of the array
            }
            if ((x >> log_2p) != ((x + k) >> log_2p)) {
                return; // not part of this odd-even merge
            }

            compare_exchange(keys, vals, u32(x), u32(x + k));
        }
    };

    template<class Tkey, class Tval>
    void sort_by_key_batcher_odd_even(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        if (len < 2) {
            return; // nothing to do, and the network below is empty anyway
        }

        using B = OddEvenOrderingPrimitive<Tkey, Tval>;

        u64 n = len;

        // each thread carries at most one comparator, and the low indices of the comparators
        // of a stage are spread over the array with a period of two
        u64 n_threads = (n + 1) / 2;

        // the loop counters are carried as exponents so that no shift can overflow when len
        // is close to the largest u32
        for (u32 log_p = 0; (u64(1) << log_p) < n; log_p++) {
            for (i32 log_k = i32(log_p); log_k >= 0; log_k--) {

                u64 k      = u64(1) << log_k;
                u64 j0     = (log_k == i32(log_p)) ? 0 : k; // k mod p
                u32 log_2p = log_p + 1;
                u32 lk     = u32(log_k);

                sham::kernel_call_u64(
                    sched->get_queue(),
                    sham::MultiRef{},
                    sham::MultiRef{buf_key, buf_values},
                    n_threads,
                    [=](u64 gid, Tkey *keys, Tval *vals) {
                        B::merge_step(keys, vals, n, k, j0, lk, log_2p, gid);
                    });
            }
        }
    }

    template<class Tkey, class Tval>
    void sort_by_key_batcher_odd_even_host_reference(
        std::vector<Tkey> &keys, std::vector<Tval> &values) {

        if (keys.size() != values.size()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the keys and the values must have the same length");
        }

        // Batcher's odd-even merge network, kept as the plain four loops on purpose, this is
        // the readable statement of what the device kernel computes.
        //
        //   for p = 1,2,4,... while p<n
        //     for k = p,p/2,...,1
        //       for j = k mod p to n-1-k step 2k
        //         for i = 0 to min(k-1, n-j-k-1)
        //           if floor((i+j)/2p) == floor((i+j+k)/2p):
        //             compare_exchange(a[i+j], a[i+j+k])

        i32 n = static_cast<i32>(keys.size());
        for (i32 p = 1; p < n; p <<= 1) {
            for (i32 k = p; k >= 1; k >>= 1) {
                for (i32 j = k % p; j <= n - 1 - k; j += 2 * k) {
                    i32 imax = std::min(k - 1, n - j - k - 1);
                    for (i32 i = 0; i <= imax; ++i) {
                        i32 idx1 = i + j;
                        i32 idx2 = i + j + k;
                        if ((idx1 / (2 * p)) == (idx2 / (2 * p))) {
                            if (keys[idx2] < keys[idx1]) {
                                std::swap(keys[idx1], keys[idx2]);
                                std::swap(values[idx1], values[idx2]);
                            }
                        }
                    }
                }
            }
        }
    }

    template void sort_by_key_batcher_odd_even<u32, u32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even<u64, u32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<u64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even<f32, f32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<f32> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even<f64, f64>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<f64> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even<f32, u32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f32> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even<f64, u32>(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<f64> &buf_key,
        sham::DeviceBuffer<u32> &buf_values,
        u32 len);

    template void sort_by_key_batcher_odd_even_host_reference<u32, u32>(
        std::vector<u32> &keys, std::vector<u32> &values);

    template void sort_by_key_batcher_odd_even_host_reference<u64, u32>(
        std::vector<u64> &keys, std::vector<u32> &values);

    template void sort_by_key_batcher_odd_even_host_reference<f32, f32>(
        std::vector<f32> &keys, std::vector<f32> &values);

    template void sort_by_key_batcher_odd_even_host_reference<f32, u32>(
        std::vector<f32> &keys, std::vector<u32> &values);

    template void sort_by_key_batcher_odd_even_host_reference<f64, u32>(
        std::vector<f64> &keys, std::vector<u32> &values);

    template void sort_by_key_batcher_odd_even_host_reference<f64, f64>(
        std::vector<f64> &keys, std::vector<f64> &values);

} // namespace shamalgs::algorithm::details
