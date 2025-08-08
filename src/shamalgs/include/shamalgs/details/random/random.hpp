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
 * @file random.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <shambackends/sycl.hpp>
#include <random>

/**
 * @brief namespace to contain utility related to random number generation in shamalgs
 *
 */
namespace shamalgs::random {

    template<class T>
    T mock_gaussian(std::mt19937 &eng) {
        using namespace shambase::constants;

        constexpr T _2pi = pi<T> * 2;
        T r_3            = shamalgs::mock_value<T>(eng, 0, 1);
        T r_4            = shamalgs::mock_value<T>(eng, 0, 1);
        return sycl::sqrt(-2 * sycl::log(r_3)) * sycl::cos(_2pi * r_4);
    }

    template<class T>
    T mock_gaussian_multidim(std::mt19937 &eng) {
        T ret;
        constexpr int n = shambase::VectorProperties<T>::dimension;
        using Tscal     = shambase::VecComponent<T>;
#pragma unroll
        for (int i = 0; i < n; i++) {
            ret[i] = mock_gaussian<Tscal>(eng);
        }
        return ret;
    }

    template<class T>
    T mock_unit_vector(std::mt19937 &eng) {
        T ret    = mock_gaussian_multidim<T>(eng);
        auto len = sycl::length(ret);

        auto default_unit_vec = []() {
            T ret  = {};
            ret[0] = 1;
            return ret;
        };

        return (len > 0) ? (ret / len) : default_unit_vec();
    }

    template<class T>
    sycl::buffer<T> mock_buffer(u64 seed, u32 len, T min_bound, T max_bound);

    template<class T>
    sham::DeviceBuffer<T> mock_buffer_usm(
        const sham::DeviceScheduler_ptr &sched, u64 seed, u32 len, T min_bound, T max_bound);

    template<class T>
    inline sham::DeviceBuffer<T>
    mock_buffer_usm(const sham::DeviceScheduler_ptr &sched, u64 seed, u32 len) {
        using Prop = shambase::VectorProperties<T>;
        return mock_buffer_usm(sched, seed, len, Prop::get_min(), Prop::get_max());
    }

    template<class T>
    inline sycl::buffer<T> mock_buffer(u64 seed, u32 len) {
        using Prop = shambase::VectorProperties<T>;
        return mock_buffer(seed, len, Prop::get_min(), Prop::get_max());
    }

    template<class T>
    inline std::unique_ptr<sycl::buffer<T>>
    mock_buffer_ptr(u64 seed, u32 len, T min_bound, T max_bound) {
        return std::make_unique<sycl::buffer<T>>(mock_buffer(seed, len, min_bound, max_bound));
    }
    template<class T>
    inline std::unique_ptr<sycl::buffer<T>> mock_buffer_ptr(u64 seed, u32 len) {
        using Prop = shambase::VectorProperties<T>;
        return mock_buffer_ptr(seed, len, Prop::get_min(), Prop::get_max());
    }

    template<class T>
    T next_obj(std::mt19937 &eng, std::uniform_real_distribution<f64> &distval);

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

} // namespace shamalgs::random
