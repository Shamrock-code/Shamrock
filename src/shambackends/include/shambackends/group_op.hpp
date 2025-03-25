// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file group_op.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace sham {

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
    template<class T, class Func>
    inline T map_vector(T in, Func &&f) {

        static constexpr u32 dim = shambase::VectorProperties<T>::dimension;

        if constexpr (dim == 1) {
            return f(in);
        } else if constexpr (dim == 2) {
            return {f(in[0]), f(in[1])};
        } else if constexpr (dim == 3) {
            return {f(in[0]), f(in[1]), f(in[2])};
        } else if constexpr (dim == 4) {
            return {f(in[0]), f(in[1]), f(in[2]), f(in[3])};
        } else if constexpr (dim == 8) {
            return {f(in[0]), f(in[1]), f(in[2]), f(in[3]), f(in[4]), f(in[5]), f(in[6]), f(in[7])};
        } else if constexpr (dim == 16) {
            return {
                f(in[0]),
                f(in[1]),
                f(in[2]),
                f(in[3]),
                f(in[4]),
                f(in[5]),
                f(in[6]),
                f(in[7]),
                f(in[8]),
                f(in[9]),
                f(in[10]),
                f(in[11]),
                f(in[12]),
                f(in[13]),
                f(in[14]),
                f(in[15])};
        } else {
            static_assert(shambase::always_false_v<decltype(dim)>, "non-exhaustive visitor!");
        }
    }

    template<class T>
    inline T sum_over_group(sycl::group<1> g, T v) {
    #ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::plus<>());
    #endif
    #ifdef SYCL_COMP_ACPP
        return map_vector(v, [&](auto component) {
            return sycl::reduce_over_group(g, component, sycl::plus<decltype(component)>{});
        });
    #endif
    }

    template<class T>
    inline T min_over_group(sycl::group<1> g, T v) {
    #ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::minimum<>());
    #endif
    #ifdef SYCL_COMP_ACPP
        return map_vector(v, [&](auto component) {
            return sycl::reduce_over_group(g, component, sycl::minimum<decltype(component)>{});
        });
    #endif
    }

    template<class T>
    inline T max_over_group(sycl::group<1> g, T v) {
    #ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::maximum<>());
    #endif
    #ifdef SYCL_COMP_ACPP
        return map_vector(v, [&](auto component) {
            return sycl::reduce_over_group(g, component, sycl::maximum<decltype(component)>{});
        });
    #endif
    }
#endif

} // namespace sham
