// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file mock_value.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the mock_value function
 */

#include "shamalgs/primitives/mock_value.hpp"
#include <random>

namespace shamalgs::primitives {

    template<>
    u8 mock_value(std::mt19937 &eng, u8 min_bound, u8 max_bound) {
        std::uniform_int_distribution<u8> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    u16 mock_value(std::mt19937 &eng, u16 min_bound, u16 max_bound) {
        std::uniform_int_distribution<u16> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    u32 mock_value(std::mt19937 &eng, u32 min_bound, u32 max_bound) {
        std::uniform_int_distribution<u32> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    u64 mock_value(std::mt19937 &eng, u64 min_bound, u64 max_bound) {
        std::uniform_int_distribution<u64> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    i8 mock_value(std::mt19937 &eng, i8 min_bound, i8 max_bound) {
        std::uniform_int_distribution<i8> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    i16 mock_value(std::mt19937 &eng, i16 min_bound, i16 max_bound) {
        std::uniform_int_distribution<i16> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    i32 mock_value(std::mt19937 &eng, i32 min_bound, i32 max_bound) {
        std::uniform_int_distribution<i32> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    i64 mock_value(std::mt19937 &eng, i64 min_bound, i64 max_bound) {
        std::uniform_int_distribution<i64> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    f32 mock_value(std::mt19937 &eng, f32 min_bound, f32 max_bound) {
        std::uniform_real_distribution<f32> dist{min_bound, max_bound};
        return dist(eng);
    }

    template<>
    f64 mock_value(std::mt19937 &eng, f64 min_bound, f64 max_bound) {
        std::uniform_real_distribution<f64> dist{min_bound, max_bound};
        return dist(eng);
    }

#define X2(_arg_)                                                                                  \
    template<>                                                                                     \
    sycl::vec<_arg_, 2> mock_value(                                                                \
        std::mt19937 &eng, sycl::vec<_arg_, 2> min_bound, sycl::vec<_arg_, 2> max_bound) {         \
        return {                                                                                   \
            mock_value(eng, min_bound.x(), max_bound.x()),                                         \
            mock_value(eng, min_bound.y(), max_bound.y())};                                        \
    }

#define X3(_arg_)                                                                                  \
    template<>                                                                                     \
    sycl::vec<_arg_, 3> mock_value(                                                                \
        std::mt19937 &eng, sycl::vec<_arg_, 3> min_bound, sycl::vec<_arg_, 3> max_bound) {         \
        return {                                                                                   \
            mock_value(eng, min_bound.x(), max_bound.x()),                                         \
            mock_value(eng, min_bound.y(), max_bound.y()),                                         \
            mock_value(eng, min_bound.z(), max_bound.z())};                                        \
    }

#define X4(_arg_)                                                                                  \
    template<>                                                                                     \
    sycl::vec<_arg_, 4> mock_value(                                                                \
        std::mt19937 &eng, sycl::vec<_arg_, 4> min_bound, sycl::vec<_arg_, 4> max_bound) {         \
        return {                                                                                   \
            mock_value(eng, min_bound.x(), max_bound.x()),                                         \
            mock_value(eng, min_bound.y(), max_bound.y()),                                         \
            mock_value(eng, min_bound.z(), max_bound.z()),                                         \
            mock_value(eng, min_bound.w(), max_bound.w())};                                        \
    }

#define X8(_arg_)                                                                                  \
    template<>                                                                                     \
    sycl::vec<_arg_, 8> mock_value(                                                                \
        std::mt19937 &eng, sycl::vec<_arg_, 8> min_bound, sycl::vec<_arg_, 8> max_bound) {         \
        return {                                                                                   \
            mock_value(eng, min_bound.s0(), max_bound.s0()),                                       \
            mock_value(eng, min_bound.s1(), max_bound.s1()),                                       \
            mock_value(eng, min_bound.s2(), max_bound.s2()),                                       \
            mock_value(eng, min_bound.s3(), max_bound.s3()),                                       \
            mock_value(eng, min_bound.s4(), max_bound.s4()),                                       \
            mock_value(eng, min_bound.s5(), max_bound.s5()),                                       \
            mock_value(eng, min_bound.s6(), max_bound.s6()),                                       \
            mock_value(eng, min_bound.s7(), max_bound.s7())};                                      \
    }

#define X16(_arg_)                                                                                 \
    template<>                                                                                     \
    sycl::vec<_arg_, 16> mock_value(                                                               \
        std::mt19937 &eng, sycl::vec<_arg_, 16> min_bound, sycl::vec<_arg_, 16> max_bound) {       \
        return {                                                                                   \
            mock_value(eng, min_bound.s0(), max_bound.s0()),                                       \
            mock_value(eng, min_bound.s1(), max_bound.s1()),                                       \
            mock_value(eng, min_bound.s2(), max_bound.s2()),                                       \
            mock_value(eng, min_bound.s3(), max_bound.s3()),                                       \
            mock_value(eng, min_bound.s4(), max_bound.s4()),                                       \
            mock_value(eng, min_bound.s5(), max_bound.s5()),                                       \
            mock_value(eng, min_bound.s6(), max_bound.s6()),                                       \
            mock_value(eng, min_bound.s7(), max_bound.s7()),                                       \
            mock_value(eng, min_bound.s8(), max_bound.s8()),                                       \
            mock_value(eng, min_bound.s9(), max_bound.s9()),                                       \
            mock_value(eng, min_bound.sA(), max_bound.sA()),                                       \
            mock_value(eng, min_bound.sB(), max_bound.sB()),                                       \
            mock_value(eng, min_bound.sC(), max_bound.sC()),                                       \
            mock_value(eng, min_bound.sD(), max_bound.sD()),                                       \
            mock_value(eng, min_bound.sE(), max_bound.sE()),                                       \
            mock_value(eng, min_bound.sF(), max_bound.sF())};                                      \
    }

#define X(_arg_) X2(_arg_) X3(_arg_) X4(_arg_) X8(_arg_) X16(_arg_)

    X(f32);
    X(f64);
    X(u32);
    X(u64);
    X(i64);

#undef X
#undef X2
#undef X3
#undef X4
#undef X8
#undef X16

} // namespace shamalgs::primitives
