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
 * @file sort_by_keys_radix_chunked.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Portable LSD radix sort by keys, parallelized over chunks of the input.
 *
 * Each pass sorts on a digit of `radix_bits` bits :
 *  1. every work-item computes the digit histogram of its contiguous chunk of the input,
 *  2. an exclusive scan of the histograms (stored bucket major, i.e. index
 *     `bucket * nchunks + chunk`) gives the output offset of every (bucket, chunk) pair,
 *  3. every work-item scatters its chunk, in order, to those offsets.
 *
 * The sort is stable. It only uses plain range kernels (no work-group primitives), so it runs on
 * any SYCL backend; it is intended for CPU-like devices where the work-items are few and fat.
 */

#include "shambase/aliases_int.hpp"
#include "shamalgs/primitives/scan_exclusive_sum_in_place.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include <type_traits>
#include <algorithm>
#include <utility>

namespace shamalgs::primitives::device::details {

    /// Stable LSD radix sort of (keys, values) on the first `len` elements (unsigned keys only)
    template<class Tkey, class Tval>
    inline void sort_by_keys_radix_chunked(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<Tval> &buf_values,
        u32 len) {

        static_assert(
            std::is_unsigned_v<Tkey>, "the radix sort is only implemented for unsigned keys");

        constexpr u32 radix_bits = 8;
        constexpr u32 nbuckets   = 1u << radix_bits;
        constexpr u32 digit_mask = nbuckets - 1;
        constexpr u32 npasses    = (sizeof(Tkey) * 8) / radix_bits;
        static_assert(npasses % 2 == 0, "the result is expected in the input buffers");

        if (len <= 1) {
            return;
        }

        // chunks of at least 4096 elements, and at most 4096 chunks
        constexpr u32 min_chunk_size = 4096;
        constexpr u32 max_chunks     = 4096;
        u32 nchunks                  = std::max(1u, std::min(max_chunks, len / min_chunk_size));
        u32 chunk_size               = (len + nchunks - 1) / nchunks;
        nchunks                      = (len + chunk_size - 1) / chunk_size;

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<Tkey> key_tmp(len, sched);
        sham::DeviceBuffer<Tval> val_tmp(len, sched);
        sham::DeviceBuffer<u32> offsets(nbuckets * nchunks, sched);

        sham::DeviceBuffer<Tkey> *key_in  = &buf_key;
        sham::DeviceBuffer<Tkey> *key_out = &key_tmp;
        sham::DeviceBuffer<Tval> *val_in  = &buf_values;
        sham::DeviceBuffer<Tval> *val_out = &val_tmp;

        for (u32 pass = 0; pass < npasses; pass++) {
            u32 shift = pass * radix_bits;

            // 1. per chunk digit histograms
            sham::kernel_call(
                q,
                sham::MultiRef{*key_in},
                sham::MultiRef{offsets},
                nchunks,
                [len, nchunks, chunk_size, shift](
                    u32 ichunk, const Tkey *__restrict keys, u32 *__restrict hist) {
                    u32 local_hist[nbuckets];
                    for (u32 b = 0; b < nbuckets; b++) {
                        local_hist[b] = 0;
                    }

                    u32 start = ichunk * chunk_size;
                    u32 end   = sham::min(start + chunk_size, len);
                    for (u32 i = start; i < end; i++) {
                        local_hist[(keys[i] >> shift) & digit_mask]++;
                    }

                    for (u32 b = 0; b < nbuckets; b++) {
                        hist[b * nchunks + ichunk] = local_hist[b];
                    }
                });

            // 2. output offset of every (bucket, chunk)
            shamalgs::primitives::scan_exclusive_sum_in_place(offsets, nbuckets * nchunks);

            // 3. stable scatter
            sham::kernel_call(
                q,
                sham::MultiRef{*key_in, *val_in, offsets},
                sham::MultiRef{*key_out, *val_out},
                nchunks,
                [len, nchunks, chunk_size, shift](
                    u32 ichunk,
                    const Tkey *__restrict keys,
                    const Tval *__restrict vals,
                    const u32 *__restrict offs,
                    Tkey *__restrict keys_out,
                    Tval *__restrict vals_out) {
                    u32 local_offs[nbuckets];
                    for (u32 b = 0; b < nbuckets; b++) {
                        local_offs[b] = offs[b * nchunks + ichunk];
                    }

                    u32 start = ichunk * chunk_size;
                    u32 end   = sham::min(start + chunk_size, len);
                    for (u32 i = start; i < end; i++) {
                        Tkey k        = keys[i];
                        u32 dst       = local_offs[(k >> shift) & digit_mask]++;
                        keys_out[dst] = k;
                        vals_out[dst] = vals[i];
                    }
                });

            std::swap(key_in, key_out);
            std::swap(val_in, val_out);
        }
    }

} // namespace shamalgs::primitives::device::details
