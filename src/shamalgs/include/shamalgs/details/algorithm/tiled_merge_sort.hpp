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
 * @file tiled_merge_sort.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tiled merge sort for key/value buffers of any length
 *
 * The sort runs in two phases:
 *
 * 1. every work item sorts a tile of `Vt` consecutive elements with a fully unrolled odd-even
 *    transpose network, held in thread private arrays,
 * 2. adjacent sorted runs are then merged pairwise, doubling the run length each round, until a
 *    single run covers the buffer. Each round is one kernel where a work item owns a fixed size
 *    chunk of the output and finds its input sub-ranges with a Merge Path co-rank search.
 *
 * The total work is O(n log n), and the length does not need to be a power of two.
 *
 * `Vt` is a template parameter because the tile network unrolls at compile time. It is a
 * O(Vt^2) comparison network, so it is only meant to be used with small tiles (order 8).
 */

#include "shambase/alg_primitives.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/numeric_limits.hpp"
#include "shambase/ptr.hpp"
#include "shamalgs/details/algorithm/tiled_merge_sort_host.hpp"
#include "shamalgs/primitives/co_rank.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include <stdexcept>

namespace shamalgs::algorithm::details {

    /**
     * @brief Sort every tile of `Vt` consecutive elements of the buffers, in place
     *
     * One work item per tile. The tile is loaded into thread private arrays, padded up to `Vt`
     * with `shambase::get_max<Tkey>()` when it is the trailing partial tile, sorted with the
     * odd-even transpose network, then the valid part is stored back.
     *
     * Since the network only swaps on a strictly smaller key, the padding never overtakes a
     * real key that compares equal to the sentinel, so the store back of the first `cnt`
     * elements stays correct even then.
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @tparam Vt Tile size, in elements per work item
     * @param sched The device scheduler
     * @param buf_key Buffer holding the keys
     * @param buf_values Buffer holding the values
     * @param len Number of elements to sort
     */
    template<class Tkey, class Tval, u32 Vt>
    inline void tiled_merge_sort_tile_pass(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        u32 n_tiles = u32((u64(len) + Vt - 1) / Vt);

        sham::kernel_call(
            shambase::get_check_ref(sched).get_queue(),
            sham::MultiRef{},
            sham::MultiRef{buf_key, buf_values},
            n_tiles,
            [len](u32 tid, Tkey *__restrict__ keys, Tval *__restrict__ vals) {
                Tkey local_keys[Vt];
                Tval local_vals[Vt];

                u32 start = tid * Vt;
                u32 cnt   = (start + Vt < len) ? Vt : (len - start);

                for (u32 i = 0; i < Vt; ++i) {
                    local_keys[i] = (i < cnt) ? keys[start + i] : shambase::get_max<Tkey>();
                    local_vals[i] = (i < cnt) ? vals[start + i] : Tval{};
                }

                shambase::odd_even_transpose_sort_by_key<static_cast<int>(Vt)>(
                    local_keys, local_vals, [](Tkey a, Tkey b) {
                        return a < b;
                    });

                for (u32 i = 0; i < cnt; ++i) {
                    keys[start + i] = local_keys[i];
                    vals[start + i] = local_vals[i];
                }
            });
    }

    /**
     * @brief Merge adjacent sorted runs of length `r` into runs of length `2 * r`
     *
     * Merging is not done in place: the result is written to a second pair of buffers. The
     * output of pair `p` is the range `[2 * p * r, min((2 * p + 2) * r, len))`, split into
     * chunks of `elems_per_thread` elements, one work item each. A work item co-ranks the
     * start and the end of its chunk to get its own sub-ranges of the two runs, then merges
     * them sequentially.
     *
     * A trailing run without a partner (odd run count) and a pair whose second run is empty
     * are both copied through unchanged.
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @param sched The device scheduler
     * @param src_key Sorted runs to read the keys from
     * @param src_val Values matching `src_key`
     * @param dst_key Buffer receiving the merged keys
     * @param dst_val Buffer receiving the merged values
     * @param len Number of elements to sort
     * @param r Current run length
     * @param elems_per_thread Number of output elements handled by one work item
     */
    template<class Tkey, class Tval>
    inline void tiled_merge_sort_merge_round(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &src_key,
        sham::DeviceBuffer<Tval> &src_val,
        sham::DeviceBuffer<Tkey> &dst_key,
        sham::DeviceBuffer<Tval> &dst_val,
        u32 len,
        u32 r,
        u32 elems_per_thread) {

        u32 n_runs  = u32((u64(len) + r - 1) / r);
        u32 n_pairs = (n_runs + 1) / 2;

        // a pair spans at most 2 * r output elements, so this is an upper bound on the chunk
        // count of any pair. Threads landing past the end of a shorter pair return immediately.
        u32 chunks_per_pair = u32((u64(2) * r + elems_per_thread - 1) / elems_per_thread);

        u64 n_threads = u64(n_pairs) * chunks_per_pair;

        sham::kernel_call_u64(
            shambase::get_check_ref(sched).get_queue(),
            sham::MultiRef{src_key, src_val},
            sham::MultiRef{dst_key, dst_val},
            n_threads,
            [len, r, elems_per_thread, chunks_per_pair](
                u64 gid,
                const Tkey *__restrict__ src_k,
                const Tval *__restrict__ src_v,
                Tkey *__restrict__ dst_k,
                Tval *__restrict__ dst_v) {
                u32 pair  = u32(gid / chunks_per_pair);
                u32 chunk = u32(gid % chunks_per_pair);

                u64 a_begin_64 = u64(2) * pair * r;
                if (a_begin_64 >= len) {
                    return;
                }
                u32 a_begin = u32(a_begin_64);

                u32 a_end   = (u64(a_begin) + r < len) ? (a_begin + r) : len;
                u32 b_begin = a_end;
                u32 b_end   = (u64(b_begin) + r < len) ? (b_begin + r) : len;

                u32 m     = a_end - a_begin;
                u32 n_b   = b_end - b_begin;
                u32 total = m + n_b;

                u64 k0_64 = u64(chunk) * elems_per_thread;
                if (k0_64 >= total) {
                    return;
                }
                u32 k0 = u32(k0_64);
                u32 k1 = (u64(k0) + elems_per_thread < total) ? (k0 + elems_per_thread) : total;

                const Tkey *a_keys = src_k + a_begin;
                const Tval *a_vals = src_v + a_begin;
                const Tkey *b_keys = src_k + b_begin;
                const Tval *b_vals = src_v + b_begin;

                u32 i0, i1, j0, j1;
                if (n_b == 0) {
                    // unpaired trailing run, or a pair whose second run is empty : copy through
                    i0 = k0;
                    i1 = k1;
                    j0 = 0;
                    j1 = 0;
                } else {
                    i0 = shamalgs::primitives::co_rank(k0, a_keys, m, b_keys, n_b);
                    j0 = k0 - i0;
                    i1 = shamalgs::primitives::co_rank(k1, a_keys, m, b_keys, n_b);
                    j1 = k1 - i1;
                }

                u32 ia = i0;
                u32 jb = j0;
                u32 o  = a_begin + k0;

                while (ia < i1 && jb < j1) {
                    // take from the first run on ties, to match the co_rank tie breaking
                    if (!(b_keys[jb] < a_keys[ia])) {
                        dst_k[o] = a_keys[ia];
                        dst_v[o] = a_vals[ia];
                        ++ia;
                    } else {
                        dst_k[o] = b_keys[jb];
                        dst_v[o] = b_vals[jb];
                        ++jb;
                    }
                    ++o;
                }
                while (ia < i1) {
                    dst_k[o] = a_keys[ia];
                    dst_v[o] = a_vals[ia];
                    ++ia;
                    ++o;
                }
                while (jb < j1) {
                    dst_k[o] = b_keys[jb];
                    dst_v[o] = b_vals[jb];
                    ++jb;
                    ++o;
                }
            });
    }

} // namespace shamalgs::algorithm::details

namespace shamalgs::algorithm {

    /**
     * @brief Sort key-value pairs on the device with a tiled merge sort
     *
     * Sorts the first `len` elements of the buffers in place, reordering the values along with
     * their keys. Unlike `sort_by_key_bitonic_updated_usm`, `len` does not need to be a power
     * of two.
     *
     * The sort is stable, but callers should not rely on it: the public `sort_by_keys` contract
     * does not promise stability.
     *
     * @tparam Tkey Key type - must be comparable (supports < and > operators)
     * @tparam Tval Value type - can be any copyable type
     * @tparam Vt Tile size, in elements per work item. O(Vt^2) network, keep it small.
     * @param sched The device scheduler
     * @param buf_key Device buffer containing the keys to sort by
     * @param buf_values Device buffer containing the values to reorder
     * @param len Number of elements to sort, at most the size of both buffers
     * @param elems_per_thread Number of output elements handled by one work item in a merge
     * round
     */
    template<class Tkey, class Tval, u32 Vt = 8>
    inline void sort_by_key_tiled_merge(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len,
        u32 elems_per_thread = 256) {

        static_assert(Vt >= 2, "the tile size must be at least 2");

        if (elems_per_thread == 0) {
            shambase::throw_with_loc<std::invalid_argument>("elems_per_thread must not be 0");
        }

        if (len > buf_key.get_size() || len > buf_values.get_size()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "len is larger than the buffers to sort");
        }

        if (len < 2) {
            return;
        }

        details::tiled_merge_sort_tile_pass<Tkey, Tval, Vt>(sched, buf_key, buf_values, len);

        if (u64(Vt) >= len) {
            // a single tile already covers the buffer, no merge round to run
            return;
        }

        sham::DeviceBuffer<Tkey> scratch_key(len, sched);
        sham::DeviceBuffer<Tval> scratch_val(len, sched);

        // ping-pong between the caller's buffers and the scratch pair by swapping host side
        // pointers, the buffers themselves are never moved
        sham::DeviceBuffer<Tkey> *src_k = &buf_key;
        sham::DeviceBuffer<Tkey> *dst_k = &scratch_key;
        sham::DeviceBuffer<Tval> *src_v = &buf_values;
        sham::DeviceBuffer<Tval> *dst_v = &scratch_val;

        for (u64 r = Vt; r < len; r *= 2) {
            details::tiled_merge_sort_merge_round(
                sched, *src_k, *src_v, *dst_k, *dst_v, len, u32(r), elems_per_thread);

            std::swap(src_k, dst_k);
            std::swap(src_v, dst_v);
        }

        if (src_k != &buf_key) {
            // odd round count, the sorted result ended up in the scratch pair
            buf_key.copy_from(scratch_key, len);
            buf_values.copy_from(scratch_val, len);
        }
    }

} // namespace shamalgs::algorithm
