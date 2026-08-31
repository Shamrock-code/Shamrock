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
 * @file tiled_merge_sort_host.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Host serial reference implementation of the tiled merge sort
 *
 * Kept free of any SYCL dependency so that the algorithm can be exercised, and its arithmetic
 * checked, without a device. See tiled_merge_sort.hpp for the device implementation.
 */

#include "shambase/alg_primitives.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/numeric_limits.hpp"
#include "shamalgs/primitives/co_rank.hpp"
#include <stdexcept>
#include <vector>

namespace shamalgs::algorithm {

    /**
     * @brief Host serial reference implementation of `sort_by_key_tiled_merge`
     *
     * Computes exactly what the device version computes, with the same tiling, the same
     * sentinel padding, the same co-rank splits and the same chunk to work item mapping, but
     * with the kernels spelled as plain loops. Kept as a readable statement of the algorithm
     * and as a differential test target.
     *
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @tparam Vt Tile size, in elements per work item
     * @param keys The keys to sort by, sorted in place
     * @param vals The values to reorder, permuted alongside the keys
     * @param elems_per_thread Number of output elements handled by one work item in a merge
     * round
     */
    template<class Tkey, class Tval, u32 Vt = 8>
    inline void tiled_merge_sort_host_serial(
        std::vector<Tkey> &keys, std::vector<Tval> &vals, u32 elems_per_thread = 256) {

        static_assert(Vt >= 2, "the tile size must be at least 2");

        if (keys.size() != vals.size()) {
            shambase::throw_with_loc<std::invalid_argument>(
                "the keys and the values must have the same length");
        }

        if (elems_per_thread == 0) {
            shambase::throw_with_loc<std::invalid_argument>("elems_per_thread must not be 0");
        }

        u32 len = u32(keys.size());

        if (len < 2) {
            return;
        }

        u32 n_tiles = u32((u64(len) + Vt - 1) / Vt);
        for (u32 tid = 0; tid < n_tiles; ++tid) {
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
        }

        if (u64(Vt) >= len) {
            return;
        }

        std::vector<Tkey> scratch_key(len);
        std::vector<Tval> scratch_val(len);

        std::vector<Tkey> *src_k = &keys;
        std::vector<Tkey> *dst_k = &scratch_key;
        std::vector<Tval> *src_v = &vals;
        std::vector<Tval> *dst_v = &scratch_val;

        for (u64 r64 = Vt; r64 < len; r64 *= 2) {
            u32 r = u32(r64);

            u32 n_runs          = u32((u64(len) + r - 1) / r);
            u32 n_pairs         = (n_runs + 1) / 2;
            u32 chunks_per_pair = u32((u64(2) * r + elems_per_thread - 1) / elems_per_thread);

            const Tkey *src_key_ptr = src_k->data();
            const Tval *src_val_ptr = src_v->data();
            Tkey *dst_key_ptr       = dst_k->data();
            Tval *dst_val_ptr       = dst_v->data();

            for (u64 gid = 0; gid < u64(n_pairs) * chunks_per_pair; ++gid) {
                u32 pair  = u32(gid / chunks_per_pair);
                u32 chunk = u32(gid % chunks_per_pair);

                u64 a_begin_64 = u64(2) * pair * r;
                if (a_begin_64 >= len) {
                    continue;
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
                    continue;
                }
                u32 k0 = u32(k0_64);
                u32 k1 = (u64(k0) + elems_per_thread < total) ? (k0 + elems_per_thread) : total;

                const Tkey *a_keys = src_key_ptr + a_begin;
                const Tval *a_vals = src_val_ptr + a_begin;
                const Tkey *b_keys = src_key_ptr + b_begin;
                const Tval *b_vals = src_val_ptr + b_begin;

                u32 i0, i1, j0, j1;
                if (n_b == 0) {
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
                    if (!(b_keys[jb] < a_keys[ia])) {
                        dst_key_ptr[o] = a_keys[ia];
                        dst_val_ptr[o] = a_vals[ia];
                        ++ia;
                    } else {
                        dst_key_ptr[o] = b_keys[jb];
                        dst_val_ptr[o] = b_vals[jb];
                        ++jb;
                    }
                    ++o;
                }
                while (ia < i1) {
                    dst_key_ptr[o] = a_keys[ia];
                    dst_val_ptr[o] = a_vals[ia];
                    ++ia;
                    ++o;
                }
                while (jb < j1) {
                    dst_key_ptr[o] = b_keys[jb];
                    dst_val_ptr[o] = b_vals[jb];
                    ++jb;
                    ++o;
                }
            }

            std::swap(src_k, dst_k);
            std::swap(src_v, dst_v);
        }

        if (src_k != &keys) {
            keys = scratch_key;
            vals = scratch_val;
        }
    }

} // namespace shamalgs::algorithm
