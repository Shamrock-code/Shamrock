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
#include "shamalgs/details/numeric/exclusiveScanAtomic.hpp"
#include "shamalgs/details/numeric/exclusiveScanGPUGems39.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/details/numeric/streamCompactExclScan.hpp"

namespace shamalgs::numeric {

    template<class T>
    sycl::buffer<T> exclusive_sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 len) {
#ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        return details::exclusive_sum_fallback(q, buf1, len);
#else
    #ifdef __HIPSYCL_ENABLE_LLVM_SSCP_TARGET__
        // SSCP does not compile decoupled lookback scan
        return details::exclusive_sum_fallback(q, buf1, len);
    #else
        return details::exclusive_sum_atomic_decoupled_v5<T, 512>(q, buf1, len);
    #endif
#endif
    }

    template<class T>
    sham::DeviceBuffer<T>
    exclusive_sum(sham::DeviceScheduler_ptr sched, sham::DeviceBuffer<T> &buf1, u32 len) {
#ifdef __MACH__ // decoupled lookback perf on mac os is awfull
        return details::exclusive_sum_fallback_usm(sched, buf1, len);
#else
    #ifdef __HIPSYCL_ENABLE_LLVM_SSCP_TARGET__
        // SSCP does not compile decoupled lookback scan
        return details::exclusive_sum_fallback_usm(sched, buf1, len);
    #else
        return details::exclusive_sum_atomic_decoupled_v5_usm<T, 512>(sched, buf1, len);
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

} // namespace shamalgs::numeric
