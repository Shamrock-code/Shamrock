// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file reduction.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/primitives/reduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/reduction.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"

namespace shamalgs::primitives {

    template<class T>
    T sum(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return shamalgs::reduction::details::sum_usm_group(sched, buf1, start_id, end_id, 128);
#else
        return shamalgs::reduction::details::sum_usm_fallback(sched, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T min(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return shamalgs::reduction::details::min_usm_group(sched, buf1, start_id, end_id, 128);
#else
        return shamalgs::reduction::details::min_usm_fallback(sched, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T max(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return shamalgs::reduction::details::max_usm_group(sched, buf1, start_id, end_id, 128);
#else
        return shamalgs::reduction::details::max_usm_fallback(sched, buf1, start_id, end_id);
#endif
    }

#ifndef DOXYGEN
    #define XMAC_TYPES                                                                             \
        X(f32)                                                                                     \
        X(f32_2)                                                                                   \
        X(f32_3)                                                                                   \
        X(f32_4)                                                                                   \
        X(f32_8)                                                                                   \
        X(f32_16)                                                                                  \
        X(f64)                                                                                     \
        X(f64_2)                                                                                   \
        X(f64_3)                                                                                   \
        X(f64_4)                                                                                   \
        X(f64_8)                                                                                   \
        X(f64_16)                                                                                  \
        X(u32)                                                                                     \
        X(u64)                                                                                     \
        X(i32)                                                                                     \
        X(i64)                                                                                     \
        X(u32_3)                                                                                   \
        X(u64_3)                                                                                   \
        X(i64_3)                                                                                   \
        X(i32_3)

    #define X(_arg_)                                                                               \
        template _arg_ sum<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ min<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ max<_arg_>(                                                                 \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);

    XMAC_TYPES
    #undef X
#endif

} // namespace shamalgs::primitives
