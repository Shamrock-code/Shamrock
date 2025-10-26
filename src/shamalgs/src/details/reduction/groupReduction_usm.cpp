// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file groupReduction_usm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction_usm_impl.hpp"
#include "shamalgs/details/reduction/group_reduc_utils.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/group_op.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/vec.hpp"
#include <stdexcept>

namespace shamalgs::reduction::details {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION

    /**
     * @brief Compute the sum of a given range in a buffer using group reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     * @param work_group_size The size of the work group to use
     *
     * @return The sum of the values in the index range
     */
    template<class T>
    T sum_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size) {

        // Empty range for sum should return 0
        if (start_id >= end_id) {
            return shambase::VectorProperties<T>::get_zero();
        }

        return reduc_internal<T>(
            sched,
            buf1,
            start_id,
            end_id,
            work_group_size,
            [](sycl::group<1> g, T v) {
                return sham::sum_over_group(g, v);
            },
            [](T lhs, T rhs) {
                return lhs + rhs;
            },
            []() {
                return shambase::VectorProperties<T>::get_zero();
            });
    }

    /**
     * @brief Compute the maximum value of a given range in a buffer using group reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     * @param work_group_size The size of the work group to use
     *
     * @return The maximum value of the range
     */
    template<class T>
    T max_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size) {

        if (start_id >= end_id) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "Empty range (or invalid range) not supported for max operation\n  start_id = {}, "
                "end_id = {}",
                start_id,
                end_id));
        }

        return reduc_internal<T>(
            sched,
            buf1,
            start_id,
            end_id,
            work_group_size,
            [](sycl::group<1> g, T v) {
                return sham::max_over_group(g, v);
            },
            [](T lhs, T rhs) {
                return sham::max(lhs, rhs);
            },
            []() {
                return shambase::VectorProperties<T>::get_min();
            });
    }

    /**
     * @brief Compute the minimum value of a given range in a buffer using group reduction
     *
     * @param sched The device scheduler to use
     * @param buf1 The buffer to read from
     * @param start_id The starting index of the range
     * @param end_id The end id of the range
     * @param work_group_size The size of the work group to use
     *
     * @return The minimum value of the range
     */
    template<class T>
    T min_usm_group(
        const sham::DeviceScheduler_ptr &sched,
        const sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id,
        u32 work_group_size) {

        if (start_id >= end_id) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "Empty range (or invalid range) not supported for min operation\n  start_id = {}, "
                "end_id = {}",
                start_id,
                end_id));
        }

        return reduc_internal<T>(
            sched,
            buf1,
            start_id,
            end_id,
            work_group_size,
            [](sycl::group<1> g, T v) {
                return sham::min_over_group(g, v);
            },
            [](T lhs, T rhs) {
                return sham::min(lhs, rhs);
            },
            []() {
                return shambase::VectorProperties<T>::get_max();
            });
    }
#endif
} // namespace shamalgs::reduction::details

#ifndef DOXYGEN

    #ifdef SYCL2020_FEATURE_GROUP_REDUCTION

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
            template _arg_ shamalgs::reduction::details::sum_usm_group<_arg_>(                     \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id,                                                                        \
                u32 work_group_size);                                                              \
            template _arg_ shamalgs::reduction::details::max_usm_group<_arg_>(                     \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id,                                                                        \
                u32 work_group_size);                                                              \
            template _arg_ shamalgs::reduction::details::min_usm_group<_arg_>(                     \
                const sham::DeviceScheduler_ptr &sched,                                            \
                const sham::DeviceBuffer<_arg_> &buf1,                                             \
                u32 start_id,                                                                      \
                u32 end_id,                                                                        \
                u32 work_group_size);

XMAC_TYPES
        #undef X

    #endif
#endif
