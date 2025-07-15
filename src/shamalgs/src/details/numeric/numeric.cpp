// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file numeric.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambase/assert.hpp"
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/details/numeric/streamCompactExclScan.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"

namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
#ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        return details::exclusive_sum_fallback(q, buf1, len);
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::exclusive_sum_atomic_decoupled_v5<T, 512>(q, buf1, len);
    #else
        return details::exclusive_sum_fallback(q, buf1, len);
    #endif
#endif
    }

    template<class T>
    sham::DeviceBuffer<T>
    exclusive_sum(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<T> &buf1, u32 len) {
#ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        return details::exclusive_sum_fallback_usm(sched, buf1, len);
#else
    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::exclusive_sum_atomic_decoupled_v5_usm<T, 512>(sched, buf1, len);
    #else
        return details::exclusive_sum_fallback_usm(sched, buf1, len);
    #endif
#endif
    }

    template<class T>
    sycl::buffer<T> inclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
        return details::inclusive_sum_fallback(q, buf1, len);
    }

    template<class T>
    void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
        buf1 = details::exclusive_sum_atomic_decoupled_v5<T, 256>(q, buf1, len);
    }

    template<class T>
    void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
        buf1 = details::inclusive_sum_fallback(q, buf1, len);
    }

    template sycl::buffer<u32> exclusive_sum(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);
    template sham::DeviceBuffer<u32>
    exclusive_sum(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<u32> &buf1, u32 len);
    template sycl::buffer<u32> inclusive_sum(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    template void exclusive_sum_in_place(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);
    template void inclusive_sum_in_place(sycl::queue &q, sycl::buffer<u32> &buf1, u32 len);

    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len) {
        return details::stream_compact_excl_scan(q, buf_flags, len);
    };

    sham::DeviceBuffer<u32> stream_compact(
        const sham::DeviceScheduler_ptr &sched, sham::DeviceBuffer<u32> &buf_flags, u32 len) {
        return details::stream_compact_excl_scan(sched, buf_flags, len);
    }

    template<class T>
    sham::DeviceBuffer<u64> device_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len) {

        SHAM_ASSERT(nbins > 1); // at least a sup and a inf
        SHAM_ASSERT(bin_edges.get_size() == nbins + 1);

        sham::DeviceBuffer<u64> counts = sham::DeviceBuffer<u64>(nbins, sched);
        counts.fill(0);

        if (len == 0) {
            return counts;
        }

        auto &q = shambase::get_check_ref(sched).get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{values, bin_edges},
            sham::MultiRef{counts},
            len,
            [nbins](
                u32 i,
                const T *__restrict values,
                const T *__restrict bin_edges,
                u64 *__restrict counts) {
                // Only count values within [bin_edges[0], bin_edges[nbins])
                if (values[i] < bin_edges[0] || values[i] >= bin_edges[nbins]) {
                    return;
                }

                u32 start_range = 0;
                u32 end_range   = nbins + 1;

                while (end_range - start_range > 1) {
                    u32 mid_range = (start_range + end_range) / 2;

                    if (values[i] < bin_edges[mid_range]) { // mid_range is a sup
                        end_range = mid_range;
                    } else { // mid_range is an inf
                        start_range = mid_range;
                    }
                }

                SHAM_ASSERT(end_range == start_range + 1);

                sycl::atomic_ref<
                    u64,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    cnt(counts[start_range]);

                cnt++;
            });

        return counts;
    }

    template sham::DeviceBuffer<u64> device_histogram<f64>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f64> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f64> &values,
        u32 len);
    template sham::DeviceBuffer<u64> device_histogram<f32>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f32> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f32> &values,
        u32 len);

} // namespace shamalgs::numeric
