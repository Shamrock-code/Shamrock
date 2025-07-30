// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file fallbackReduction_usm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction::details {

    template<class T, class BinaryOp>
    T reduc_internal(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        BinaryOp &&bop) {

        if (!(end_id > start_id)) {
            shambase::throw_unimplemented("whaaaat are you doing");
        }

        auto acc = buf1.copy_to_stdvec();
        T ret    = acc[start_id];
        for (u32 i = start_id + 1; i < end_id; i++) {
            ret = bop(ret, acc[i]);
        }
        return ret;
    }

    template<class T>
    T sum_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        return reduc_internal<T>(sched, buf1, start_id, end_id, [](T lhs, T rhs) {
            return lhs + rhs;
        });
    }

    template<class T>
    T max_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        return reduc_internal<T>(sched, buf1, start_id, end_id, [](T lhs, T rhs) {
            return sham::max(lhs, rhs);
        });
    }

    template<class T>
    T min_usm_fallback(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        return reduc_internal<T>(sched, buf1, start_id, end_id, [](T lhs, T rhs) {
            return sham::min(lhs, rhs);
        });
    }

} // namespace shamalgs::reduction::details

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
        template _arg_ shamalgs::reduction::details::sum_usm_fallback<_arg_>(                      \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ shamalgs::reduction::details::max_usm_fallback<_arg_>(                      \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);                                                                           \
        template _arg_ shamalgs::reduction::details::min_usm_fallback<_arg_>(                      \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);

XMAC_TYPES
    #undef X
#endif
