// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file digit_histogram.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the digit histogram primitive.
 */

#include "shamalgs/primitives/digit_histogram.hpp"
#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/narrowing.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl.hpp"
#include <type_traits>
#include <algorithm>

namespace shamalgs::primitives {

    template<class Tkey, u32 radix_bits>
    void digit_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<u32> &buf_hist,
        u32 len) {

        static_assert(std::is_unsigned_v<Tkey>, "the digit histogram requires unsigned keys");
        static_assert(radix_bits >= 1 && radix_bits <= 8, "radix_bits must be in [1, 8]");
        static_assert(
            (sizeof(Tkey) * 8) % radix_bits == 0, "radix_bits must divide the bit size of Tkey");

        constexpr u32 nbuckets   = 1u << radix_bits;
        constexpr u32 digit_mask = nbuckets - 1;
        constexpr u32 npasses    = (sizeof(Tkey) * 8) / radix_bits;
        constexpr u32 hist_size  = npasses * nbuckets;

        constexpr u32 group_size       = 256;
        constexpr u32 items_per_thread = 8;
        constexpr u32 chunk_size       = group_size * items_per_thread;

        using atomic_ref_local = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_work_group,
            sycl::access::address_space::local_space>;

        using atomic_ref_global = sycl::atomic_ref<
            u32,
            sycl::memory_order_relaxed,
            sycl::memory_scope_device,
            sycl::access::address_space::global_space>;

        if (len > buf_key.get_size()) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "the key buffer is smaller than the length of the histogram\n"
                "len = {}, buf_key.get_size() = {}",
                len,
                buf_key.get_size()));
        }

        buf_hist.resize(hist_size, false);
        buf_hist.fill(0);

        if (len == 0) {
            return;
        }

        u32 nchunks = shambase::group_count(len, chunk_size);

        // enough groups to fill the device, while bounding the number of global atomics
        u32 hist_groups = std::min<u32>(nchunks, 1024);

        sham::kernel_call_hndl(
            sched->get_queue(),
            sham::MultiRef{buf_key},
            sham::MultiRef{buf_hist},
            hist_groups * group_size,
            [=](u32 nthreads, const Tkey *__restrict keys, u32 *__restrict hist) {
                return [=](sycl::handler &cgh) {
                    sycl::local_accessor<u32, 1> l_hist{hist_size, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<1>{nthreads, group_size}, [=](sycl::nd_item<1> item) {
                            u32 lid   = item.get_local_id(0);
                            u32 group = item.get_group_linear_id();

                            for (u32 b = lid; b < hist_size; b += group_size) {
                                l_hist[b] = 0;
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            // grid-stride over the chunks, coalesced reads within a chunk
                            for (u32 c = group; c < nchunks; c += hist_groups) {
                                for (u32 j = 0; j < items_per_thread; j++) {
                                    u32 i = c * chunk_size + j * group_size + lid;
                                    if (i < len) {
                                        Tkey k = keys[i];
                                        for (u32 p = 0; p < npasses; p++) {
                                            u32 digit = u32(k >> (p * radix_bits)) & digit_mask;
                                            atomic_ref_local(l_hist[p * nbuckets + digit])
                                                .fetch_add(1U);
                                        }
                                    }
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            for (u32 b = lid; b < hist_size; b += group_size) {
                                u32 cnt = l_hist[b];
                                if (cnt > 0) {
                                    atomic_ref_global(hist[b]).fetch_add(cnt);
                                }
                            }
                        });
                };
            });
    }

    template<class Tkey, u32 radix_bits>
    void digit_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<Tkey> &buf_key,
        sham::DeviceBuffer<u32> &buf_hist) {
        digit_histogram<Tkey, radix_bits>(
            sched, buf_key, buf_hist, shambase::narrow_or_throw<u32>(buf_key.get_size()));
    }

#define X(Tkey, radix_bits)                                                                        \
    template void digit_histogram<Tkey, radix_bits>(                                               \
        const sham::DeviceScheduler_ptr &sched,                                                    \
        const sham::DeviceBuffer<Tkey> &buf_key,                                                   \
        sham::DeviceBuffer<u32> &buf_hist,                                                         \
        u32 len);                                                                                  \
    template void digit_histogram<Tkey, radix_bits>(                                               \
        const sham::DeviceScheduler_ptr &sched,                                                    \
        const sham::DeviceBuffer<Tkey> &buf_key,                                                   \
        sham::DeviceBuffer<u32> &buf_hist);

    X(u32, 1)
    X(u32, 2)
    X(u32, 4)
    X(u32, 8)
    X(u64, 1)
    X(u64, 2)
    X(u64, 4)
    X(u64, 8)

#undef X

} // namespace shamalgs::primitives
