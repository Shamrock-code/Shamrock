// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include "shamalgs/details/reduction/reduction.hpp"
#include "shambase/floats.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/primitives/is_all_true.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction {

    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::sum(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::sum(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::max(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::max(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::min(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::min(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    bool has_nan(sham::DeviceBuffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            auto &dev_sched = buf.get_dev_scheduler_ptr();

            // res is filled with 1 if no nan 0 otherwise
            sham::DeviceBuffer<u8> res(cnt, dev_sched);

            sham::kernel_call(
                shambase::get_check_ref(dev_sched).get_queue(),
                sham::MultiRef{buf},
                sham::MultiRef{res},
                u32(cnt),
                [](u32 i, const T *in, u8 *out) {
                    out[i] = !sham::has_nan(in[i]);
                });

            return !shamalgs::primitives::is_all_true(res, u32(cnt));
        } else {
            return false;
        }
    }

    template<class T>
    bool has_inf(sham::DeviceBuffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            auto &dev_sched = buf.get_dev_scheduler_ptr();

            // res is filled with 1 if no inf 0 otherwise
            sham::DeviceBuffer<u8> res(cnt, dev_sched);

            sham::kernel_call(
                shambase::get_check_ref(dev_sched).get_queue(),
                sham::MultiRef{buf},
                sham::MultiRef{res},
                u32(cnt),
                [](u32 i, const T *in, u8 *out) {
                    out[i] = !sham::has_inf(in[i]);
                });

            return !shamalgs::primitives::is_all_true(res, u32(cnt));
        } else {
            return false;
        }
    }

    template<class T>
    bool has_nan_or_inf(sham::DeviceBuffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            auto &dev_sched = buf.get_dev_scheduler_ptr();

            // res is filled with 1 if no nan or inf 0 otherwise
            sham::DeviceBuffer<u8> res(cnt, dev_sched);

            sham::kernel_call(
                shambase::get_check_ref(dev_sched).get_queue(),
                sham::MultiRef{buf},
                sham::MultiRef{res},
                u32(cnt),
                [](u32 i, const T *in, u8 *out) {
                    out[i] = !sham::has_nan_or_inf(in[i]);
                });

            return !shamalgs::primitives::is_all_true(res, u32(cnt));
        } else {
            return false;
        }
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
        template _arg_ sum(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template _arg_ max(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template _arg_ min(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template bool has_nan(sham::DeviceBuffer<_arg_> &buf1, u64 cnt);                           \
        template bool has_inf(sham::DeviceBuffer<_arg_> &buf1, u64 cnt);                           \
        template bool has_nan_or_inf(sham::DeviceBuffer<_arg_> &buf1, u64 cnt);

    XMAC_TYPES
    #undef X
#endif

} // namespace shamalgs::reduction
