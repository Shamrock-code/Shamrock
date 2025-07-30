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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambase/assert.hpp"
#include "shambase/integer.hpp"
#include "shambase/numeric_limits.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/details/numeric/streamCompactExclScan.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include <utility>

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

    template<class Tret, class T>
    sham::DeviceBuffer<Tret> device_histogram(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values,
        u32 len) {

        SHAM_ASSERT(nbins > 1); // at least a sup and a inf
        SHAM_ASSERT(bin_edges.get_size() == nbins + 1);

        sham::DeviceBuffer<Tret> counts = sham::DeviceBuffer<Tret>(nbins, sched);
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
                Tret *__restrict counts) {
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
                    Tret,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_device,
                    sycl::access::address_space::global_space>
                    cnt(counts[start_range]);

                cnt++;
            });

        return counts;
    }

    template sham::DeviceBuffer<u64> device_histogram<u64, f64>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f64> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f64> &values,
        u32 len);
    template sham::DeviceBuffer<u64> device_histogram<u64, f32>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f32> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f32> &values,
        u32 len);
    template sham::DeviceBuffer<u32> device_histogram<u32, f64>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f64> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f64> &values,
        u32 len);
    template sham::DeviceBuffer<u32> device_histogram<u32, f32>(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f32> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f32> &values,
        u32 len);

    template<class T>
    BinnedCompute<T> binned_init_compute(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<T> &values, // ie f(r)
        const sham::DeviceBuffer<T> &keys,   // ie r
        u32 len) {                           // ie return <f(r)>_r

        auto &q = shambase::get_check_ref(sched).get_queue();

        auto value_filter = [&]() {
            if (len > 0) {

                // filter values
                sham::DeviceBuffer<u32> key_filter(keys.get_size(), sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{keys, bin_edges},
                    sham::MultiRef{key_filter},
                    len,
                    [nbins](
                        u32 i,
                        const T *__restrict keys,
                        const T *__restrict bin_edges,
                        u32 *__restrict key_filter) {
                        // Only count keys within [bin_edges[0], bin_edges[nbins])
                        if (keys[i] < bin_edges[0] || keys[i] >= bin_edges[nbins]) {
                            key_filter[i] = 0;
                        } else {
                            key_filter[i] = 1;
                        }
                    });

                // compact
                sham::DeviceBuffer<u32> valid_key_idxs = stream_compact(sched, key_filter, len);

                return valid_key_idxs;
            } else {
                return sham::DeviceBuffer<u32>(0, sched);
            }
        };

        sham::DeviceBuffer<u32> valid_key_idxs = value_filter();

        u32 valid_key_count = valid_key_idxs.get_size();

        // make the buffer with all the valid keys
        sham::DeviceBuffer<T> valid_keys(valid_key_count, sched);
        sham::DeviceBuffer<T> valid_values(valid_key_count, sched);

        if (valid_key_count > 0) {
            sham::kernel_call(
                q,
                sham::MultiRef{keys, values, valid_key_idxs},
                sham::MultiRef{valid_keys, valid_values},
                valid_key_count,
                [](u32 i,
                   const T *__restrict keys,
                   const T *__restrict values,
                   const u32 *__restrict valid_keys_idxs,
                   T *__restrict valid_keys,
                   T *__restrict valid_values) {
                    u32 src_key     = valid_keys_idxs[i];
                    valid_keys[i]   = keys[src_key];
                    valid_values[i] = values[src_key];
                });
        }

        // histogram standard
        sham::DeviceBuffer<u32> bin_counts
            = device_histogram<u32>(sched, bin_edges, nbins, valid_keys, valid_key_count);

        bin_counts.expand(1);
        bin_counts.set_val_at_idx(bin_counts.get_size() - 1, 0);

        // exclusive scan
        // bin_ids[i] starts at offset[i] and ends at offset[i+1]
        sham::DeviceBuffer<u32> offsets_bins
            = exclusive_sum(sched, bin_counts, bin_counts.get_size());

        SHAM_ASSERT(offsets_bins.get_val_at_idx(offsets_bins.get_size() - 1) == valid_key_count);

        if (valid_key_count > 0) {
            // sort need 2^n as length
            u32 pow2_len_key = shambase::roundup_pow2(valid_key_count);
            {
                if (pow2_len_key > valid_key_count) {
                    valid_keys.resize(pow2_len_key);
                    valid_values.resize(pow2_len_key);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{},
                        sham::MultiRef{valid_keys, valid_values},
                        pow2_len_key - valid_key_count,
                        [offset_start = valid_key_count](
                            u32 i, T *__restrict valid_keys, T *__restrict valid_values) {
                            u32 key_id           = offset_start + i;
                            valid_keys[key_id]   = shambase::get_max<T>();
                            valid_values[key_id] = shambase::get_max<T>();
                        });
                }
            }

            // how to be a patate? Resize buffers to diligently become powers of 2, and don't update
            // the variable holding their length
            shamalgs::algorithm::sort_by_key(sched, valid_keys, valid_values, pow2_len_key);
        }

        return {std::move(valid_values), std::move(offsets_bins)};
    }

    template BinnedCompute<f64> binned_init_compute(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f64> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f64> &values,
        const sham::DeviceBuffer<f64> &keys,
        u32 len);
    template BinnedCompute<f32> binned_init_compute(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<f32> &bin_edges,
        u64 nbins,
        const sham::DeviceBuffer<f32> &values,
        const sham::DeviceBuffer<f32> &keys,
        u32 len);

} // namespace shamalgs::numeric
