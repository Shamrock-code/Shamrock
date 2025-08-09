// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file random.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/random/random.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/primitives/mock_vector.hpp"

namespace shamalgs::random {

    template<class T>
    sycl::buffer<T> mock_buffer(u64 seed, u32 len, T min_bound, T max_bound) {
        return shamalgs::memory::vec_to_buf(
            shamalgs::primitives::mock_vector(seed, len, min_bound, max_bound));
    }

    template<class T>
    sham::DeviceBuffer<T> mock_buffer_usm(
        const sham::DeviceScheduler_ptr &sched, u64 seed, u32 len, T min_bound, T max_bound) {
        auto vec = shamalgs::primitives::mock_vector(seed, len, min_bound, max_bound);
        sham::DeviceBuffer<T> ret(len, sched);
        ret.copy_from_stdvec(vec);
        return ret;
    }

#ifndef DOXYGEN
    #define X(_arg_)                                                                               \
        template sycl::buffer<_arg_> mock_buffer(                                                  \
            u64 seed, u32 len, _arg_ min_bound, _arg_ max_bound);                                  \
        template sham::DeviceBuffer<_arg_> mock_buffer_usm(                                        \
            const sham::DeviceScheduler_ptr &sched,                                                \
            u64 seed,                                                                              \
            u32 len,                                                                               \
            _arg_ min_bound,                                                                       \
            _arg_ max_bound);

    X(f32);
    X(f32_2);
    X(f32_3);
    X(f32_4);
    X(f32_8);
    X(f32_16);
    X(f64);
    X(f64_2);
    X(f64_3);
    X(f64_4);
    X(f64_8);
    X(f64_16);
    X(u8);
    X(u32);
    X(u32_2);
    X(u32_3);
    X(u32_4);
    X(u32_8);
    X(u32_16);
    X(u64);
    X(u64_2);
    X(u64_3);
    X(u64_4);
    X(u64_8);
    X(u64_16);
    X(i64_3);
    X(i64);
#endif

} // namespace shamalgs::random
