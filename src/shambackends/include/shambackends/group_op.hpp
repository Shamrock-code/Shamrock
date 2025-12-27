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
 * @file group_op.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include <utility>

namespace sham {

    /**
     * @brief Apply function to each component of a vector
     *
     * @tparam T Vector type
     * @tparam Func Function type
     * @tparam Args Variadic argument types
     * @param in Input vector
     * @param f Function to apply to each component
     * @param args Additional arguments to forward to the function
     * @return T Vector with function applied to each component
     */
    template<class T, class Func, class... Args>
    inline T map_vector(const T &in, Func &&f, Args... args) {

        static constexpr u32 dim = shambase::VectorProperties<T>::dimension;

        if constexpr (dim == 1) {
            return f(in, std::forward<Args>(args)...);
        } else if constexpr (dim == 2) {
            return {f(in[0], std::forward<Args>(args)...), f(in[1], std::forward<Args>(args)...)};
        } else if constexpr (dim == 3) {
            return {
                f(in[0], std::forward<Args>(args)...),
                f(in[1], std::forward<Args>(args)...),
                f(in[2], std::forward<Args>(args)...)};
        } else if constexpr (dim == 4) {
            return {
                f(in[0], std::forward<Args>(args)...),
                f(in[1], std::forward<Args>(args)...),
                f(in[2], std::forward<Args>(args)...),
                f(in[3], std::forward<Args>(args)...)};
        } else if constexpr (dim == 8) {
            return {
                f(in[0], std::forward<Args>(args)...),
                f(in[1], std::forward<Args>(args)...),
                f(in[2], std::forward<Args>(args)...),
                f(in[3], std::forward<Args>(args)...),
                f(in[4], std::forward<Args>(args)...),
                f(in[5], std::forward<Args>(args)...),
                f(in[6], std::forward<Args>(args)...),
                f(in[7], std::forward<Args>(args)...)};
        } else if constexpr (dim == 16) {
            return {
                f(in[0], std::forward<Args>(args)...),
                f(in[1], std::forward<Args>(args)...),
                f(in[2], std::forward<Args>(args)...),
                f(in[3], std::forward<Args>(args)...),
                f(in[4], std::forward<Args>(args)...),
                f(in[5], std::forward<Args>(args)...),
                f(in[6], std::forward<Args>(args)...),
                f(in[7], std::forward<Args>(args)...),
                f(in[8], std::forward<Args>(args)...),
                f(in[9], std::forward<Args>(args)...),
                f(in[10], std::forward<Args>(args)...),
                f(in[11], std::forward<Args>(args)...),
                f(in[12], std::forward<Args>(args)...),
                f(in[13], std::forward<Args>(args)...),
                f(in[14], std::forward<Args>(args)...),
                f(in[15], std::forward<Args>(args)...)};
        } else {
            static_assert(shambase::always_false_v<decltype(dim)>, "non-exhaustive visitor!");
        }
    }

    // Note here
    // Never ever capture the sycl::group<1> by reference
    // This messes up the reduction and every component will be reduced by the same value
    // I have no feaking idea why

    /**
     * @brief Sum reduction across work-group
     *
     * @tparam T Value type to reduce
     * @param g SYCL work-group
     * @param v Value to reduce
     * @return T Sum of values across all work-items in the group
     */
    template<class T>
    inline T sum_over_group(const sycl::group<1> &g, const T &v) {
#ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::plus<>());
#endif
#ifdef SYCL_COMP_ACPP
        return map_vector(
            v,
            [](auto component, const sycl::group<1> &g) {
                return sycl::reduce_over_group(g, component, sycl::plus<decltype(component)>{});
            },
            g);
#endif
    }

    /**
     * @brief Minimum reduction across work-group
     *
     * @tparam T Value type to reduce
     * @param g SYCL work-group
     * @param v Value to reduce
     * @return T Minimum value across all work-items in the group
     */
    template<class T>
    inline T min_over_group(const sycl::group<1> &g, const T &v) {
#ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::minimum<>());
#endif
#ifdef SYCL_COMP_ACPP
        return map_vector(
            v,
            [](auto component, const sycl::group<1> &g) {
                return sycl::reduce_over_group(g, component, sycl::minimum<decltype(component)>{});
            },
            g);
#endif
    }

    /**
     * @brief Maximum reduction across work-group
     *
     * @tparam T Value type to reduce
     * @param g SYCL work-group
     * @param v Value to reduce
     * @return T Maximum value across all work-items in the group
     */
    template<class T>
    inline T max_over_group(const sycl::group<1> &g, const T &v) {
#ifdef SYCL_COMP_INTEL_LLVM
        return sycl::reduce_over_group(g, v, sycl::maximum<>());
#endif
#ifdef SYCL_COMP_ACPP
        return map_vector(
            v,
            [](auto component, const sycl::group<1> &g) {
                return sycl::reduce_over_group(g, component, sycl::maximum<decltype(component)>{});
            },
            g);
#endif
    }
#endif

} // namespace sham
