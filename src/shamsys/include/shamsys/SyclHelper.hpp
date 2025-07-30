// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SyclHelper.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include <ostream>
#include <random>

namespace shamsys::syclhelper::mock {

    template<class T>
    sycl::buffer<T> mock_buffer(u32 len, std::mt19937 &eng);

} // namespace shamsys::syclhelper::mock

namespace shamsys::syclhelper {

    template<class T>
    inline void print_vec(std::ostream &ostream, T a);

    template<class T>
    inline T next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval);

    template<class T>
    inline void print_vec(std::ostream &ostream, sycl::vec<T, 2> a) {
        ostream << "(" << a.x() << "," << a.y() << ")";
    }

    template<class T>
    inline void print_vec(std::ostream &ostream, sycl::vec<T, 3> a) {
        ostream << "(" << a.x() << "," << a.y() << "," << a.z() << ")";
    }

    template<class T>
    inline void print_vec(std::ostream &ostream, sycl::vec<T, 4> a) {
        ostream << "(" << a.x() << "," << a.y() << "," << a.z() << "," << a.w() << ")";
    }

    template<class T>
    inline void print_vec(std::ostream &ostream, sycl::vec<T, 8> a) {
        ostream << "(" << a.s0() << "," << a.s1() << "," << a.s2() << "," << a.s3() << a.s4() << ","
                << a.s5() << "," << a.s6() << "," << a.s7() << ")";
    }

    template<class T>
    inline void print_vec(std::ostream &ostream, sycl::vec<T, 16> a) {
        ostream << "(" << a.s0() << "," << a.s1() << "," << a.s2() << "," << a.s3() << a.s4() << ","
                << a.s5() << "," << a.s6() << "," << a.s7() << a.s8() << "," << a.s9() << ","
                << a.sA() << "," << a.sB() << a.sC() << "," << a.sD() << "," << a.sE() << ","
                << a.sF() << ")";
    }

    template<class T>
    inline void print_vec(std::ostream &ostream, T a) {
        ostream << a;
    }

    template<>
    inline i64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i64(distval(eng));
    }
    template<>
    inline i32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i32(distval(eng));
    }
    template<>
    inline i16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i16(distval(eng));
    }
    template<>
    inline i8 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return i8(distval(eng));
    }
    template<>
    inline u64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u64(distval(eng));
    }
    template<>
    inline u32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u32(distval(eng));
    }
    template<>
    inline u16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u16(distval(eng));
    }
    template<>
    inline u8 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return u8(distval(eng));
    }
#ifdef SYCL_COMP_INTEL_LLVM
    template<>
    inline f16 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f16(distval(eng));
    }
#endif
    template<>
    inline f32 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f32(distval(eng));
    }
    template<>
    inline f64 next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return f64(distval(eng));
    }

    template<>
    inline sycl::vec<f32, 2>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 2>{next_obj<f32>(eng, distval), next_obj<f32>(eng, distval)};
    }
    template<>
    inline sycl::vec<f32, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 3>{
            next_obj<f32>(eng, distval), next_obj<f32>(eng, distval), next_obj<f32>(eng, distval)};
    }
    template<>
    inline sycl::vec<f32, 4>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 4>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f32, 8>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 8>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f32, 16>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f32, 16>{
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval),
            next_obj<f32>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 2>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 2>{next_obj<f64>(eng, distval), next_obj<f64>(eng, distval)};
    }
    template<>
    inline sycl::vec<f64, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 3>{
            next_obj<f64>(eng, distval), next_obj<f64>(eng, distval), next_obj<f64>(eng, distval)};
    }
    template<>
    inline sycl::vec<f64, 4>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 4>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 8>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 8>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<f64, 16>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<f64, 16>{
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval),
            next_obj<f64>(eng, distval)};
    }

    template<>
    inline sycl::vec<u16, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<u16, 3>{
            next_obj<u16>(eng, distval), next_obj<u16>(eng, distval), next_obj<u16>(eng, distval)};
    }

    template<>
    inline sycl::vec<u32, 3>
    next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval) {
        return sycl::vec<u32, 3>{
            next_obj<u32>(eng, distval), next_obj<u32>(eng, distval), next_obj<u32>(eng, distval)};
    }

    template<class T>
    struct get_base_sycl_type {
        using type               = T;
        static const i32 int_len = 1;
    };

    template<class T, i32 N>
    struct get_base_sycl_type<sycl::vec<T, N>> {
        using type               = T;
        static const i32 int_len = N;
    };

} // namespace shamsys::syclhelper
