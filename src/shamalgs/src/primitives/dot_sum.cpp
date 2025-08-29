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
 * @brief
 */

#include "shamalgs/primitives/dot_sum.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambase/string.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include <stdexcept>

namespace shamalgs::primitives {

    template<class T>
    shambase::VecComponent<T> dot_sum(
        const sham::DeviceScheduler_ptr &sched,
        sham::DeviceBuffer<T> &buf1,
        u32 start_id,
        u32 end_id) {

        if (start_id > end_id) {
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("start_id > end_id: {} > {}", start_id, end_id));
        }

        if (start_id == end_id) {
            return shambase::VecComponent<T>(0);
        }

        using Tscal = shambase::VecComponent<T>;

        auto &q = shambase::get_check_ref(sched).get_queue();

        sham::DeviceBuffer<Tscal> buf_scal(end_id - start_id, sched);

        sham::kernel_call(
            q,
            sham::MultiRef{buf1},
            sham::MultiRef{buf_scal},
            end_id - start_id,
            [start_id](u32 i, const T *in, Tscal *out) {
                auto a = in[i + start_id];
                out[i] = sham::dot(a, a);
            });

        // TODO: this is not optimal as in sum another buffer will be created ...
        // I should add a sum function that does not do this, maybe in place, i don't know
        return shamalgs::primitives::sum(sched, buf_scal, 0, end_id - start_id);
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
        template shambase::VecComponent<_arg_> shamalgs::primitives::dot_sum<_arg_>(               \
            const sham::DeviceScheduler_ptr &sched,                                                \
            sham::DeviceBuffer<_arg_> &buf1,                                                       \
            u32 start_id,                                                                          \
            u32 end_id);

    XMAC_TYPES

    #undef X
    #undef XMAC_TYPES

#endif

} // namespace shamalgs::primitives
