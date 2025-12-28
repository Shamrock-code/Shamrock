// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file dot_sum.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the dot_sum primitive.
 */

#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include <stdexcept>

namespace shamalgs::primitives {

    template<class T>
    shambase::VecComponent<T> dot_sum(sham::DeviceBuffer<T> &buf1, u32 start_id, u32 end_id) {
        using Tscal = shambase::VecComponent<T>;

        if (start_id == end_id) {
            return Tscal(0);
        }

        if (start_id > end_id) {
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("start_id > end_id : {} > {}", start_id, end_id));
        }

        sham::DeviceBuffer<Tscal> ret_data_base(end_id - start_id, buf1.get_dev_scheduler_ptr());

        sham::kernel_call(
            buf1.get_queue(),
            sham::MultiRef{buf1},
            sham::MultiRef{ret_data_base},
            end_id - start_id,
            [start_id](u32 i, const T *__restrict buf1, Tscal *__restrict out) {
                T in   = buf1[i + start_id];
                out[i] = sham::dot(in, in);
            });

        return shamalgs::primitives::sum(
            buf1.get_dev_scheduler_ptr(), ret_data_base, 0, end_id - start_id);
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
        template shambase::VecComponent<_arg_> dot_sum(                                            \
            sham::DeviceBuffer<_arg_> &buf1, u32 start_id, u32 end_id);

    XMAC_TYPES
    #undef X
#endif
} // namespace shamalgs::primitives
