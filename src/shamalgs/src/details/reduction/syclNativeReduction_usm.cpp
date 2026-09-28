// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file syclNativeReduction_usm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/reduction/syclNativeReduction_usm.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"
#include <stdexcept>

namespace shamalgs::reduction::details {
#ifdef SYCL2020_FEATURE_REDUCTION

    template<class T, class BinaryOp>
    T reduc_internal_sycl_native(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        T identity,
        BinaryOp &&bop) {

        sham::DeviceQueue &q = shambase::get_check_ref(sched).get_queue();

        u32 len = end_id - start_id;

        sham::DeviceBuffer<T> recov_buf(1, sched);

        sham::EventList depends_list;
        const T *in_ptr = buf1.get_read_access(depends_list) + start_id;
        T *result       = recov_buf.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            // Without initialize_to_identity, the reduction combines with whatever was
            // already in *result (uninitialized device memory here), not with identity.
            auto reduc = sycl::reduction(
                result,
                identity,
                bop,
                sycl::property_list{sycl::property::reduction::initialize_to_identity{}});

            cgh.parallel_for(sycl::range<1>{len}, reduc, [=](sycl::id<1> idx, auto &acc) {
                acc.combine(in_ptr[idx]);
            });
        });

        buf1.complete_event_state(e);
        recov_buf.complete_event_state(e);

        return recov_buf.copy_to_stdvec()[0];
    }

    /**
     * @brief Compute the sum of a given range in a buffer using sycl::reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     *
     * @return The sum of the values in the index range
     */
    template<class T>
    T sum_usm_sycl_native(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        // Empty range for sum should return 0
        if (start_id >= end_id) {
            return shambase::VectorProperties<T>::get_zero();
        }

        return reduc_internal_sycl_native<T>(
            sched,
            buf1,
            start_id,
            end_id,
            shambase::VectorProperties<T>::get_zero(),
            [](T lhs, T rhs) {
                return lhs + rhs;
            });
    }

    /**
     * @brief Compute the maximum value of a given range in a buffer using sycl::reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     *
     * @return The maximum value of the range
     */
    template<class T>
    T max_usm_sycl_native(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        if (start_id >= end_id) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "Empty range (or invalid range) not supported for max operation\n  start_id = {}, "
                "end_id = {}",
                start_id,
                end_id));
        }

        return reduc_internal_sycl_native<T>(
            sched,
            buf1,
            start_id,
            end_id,
            shambase::VectorProperties<T>::get_min(),
            [](T lhs, T rhs) {
                return sham::max(lhs, rhs);
            });
    }

    /**
     * @brief Compute the minimum value of a given range in a buffer using sycl::reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     *
     * @return The minimum value of the range
     */
    template<class T>
    T min_usm_sycl_native(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        if (start_id >= end_id) {
            shambase::throw_with_loc<std::invalid_argument>(sham::format(
                "Empty range (or invalid range) not supported for min operation\n  start_id = {}, "
                "end_id = {}",
                start_id,
                end_id));
        }

        return reduc_internal_sycl_native<T>(
            sched,
            buf1,
            start_id,
            end_id,
            shambase::VectorProperties<T>::get_max(),
            [](T lhs, T rhs) {
                return sham::min(lhs, rhs);
            });
    }

#endif
} // namespace shamalgs::reduction::details

#ifndef DOXYGEN

    #ifdef SYCL2020_FEATURE_REDUCTION

        #define XMAC_TYPES                                                                         \
            X(f32)                                                                                 \
            X(f32_2)                                                                               \
            X(f32_3)                                                                               \
            X(f32_4)                                                                               \
            X(f32_8)                                                                               \
            X(f32_16)                                                                              \
            X(f64)                                                                                 \
            X(f64_2)                                                                               \
            X(f64_3)                                                                               \
            X(f64_4)                                                                               \
            X(f64_8)                                                                               \
            X(f64_16)                                                                              \
            X(u32)                                                                                 \
            X(u64)                                                                                 \
            X(i32)                                                                                 \
            X(i64)                                                                                 \
            X(u32_3)                                                                               \
            X(u64_3)                                                                               \
            X(i64_3)                                                                               \
            X(i32_3)

        #define X(_arg_)                                                                           \
            template _arg_ shamalgs::reduction::details::sum_usm_sycl_native<_arg_>(               \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id);                                                                       \
            template _arg_ shamalgs::reduction::details::max_usm_sycl_native<_arg_>(               \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id);                                                                       \
            template _arg_ shamalgs::reduction::details::min_usm_sycl_native<_arg_>(               \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id);

XMAC_TYPES
        #undef X

    #endif
#endif
