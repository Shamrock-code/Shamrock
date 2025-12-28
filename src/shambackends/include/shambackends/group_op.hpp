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
 * @brief SYCL work-group collective operations with proper synchronization
 *
 * This file provides portable implementations of work-group reduction operations
 * that work correctly across different SYCL implementations (Intel LLVM, AdaptiveCPP).
 *
 * IMPORTANT: For vector types on AdaptiveCPP, each component must be reduced
 * separately with group barriers between reductions. This is because:
 * 1. reduce_over_group is a collective operation requiring all work-items to participate
 * 2. Calling multiple collective operations sequentially without barriers corrupts
 *    the internal synchronization state of the group object
 * 3. Intel LLVM handles vector types natively in a single collective call
 */
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include <utility>

namespace sham {

#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
    /**
     * @brief Apply function to each component of a vector
     *
     * WARNING: Do NOT use this for collective operations like reduce_over_group
     * without explicit barriers between calls. Use the specialized reduction
     * functions below instead.
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

    namespace detail {

        /**
         * @brief Reduce a single component with group barrier
         *
         * Helper to perform a single scalar reduction. The barrier ensures
         * proper synchronization between consecutive collective operations.
         */
        template<class Tscal, class Op>
        inline Tscal reduce_component(const sycl::group<1> &g, Tscal component, Op op) {
            auto result = sycl::reduce_over_group(g, component, op);
            sycl::group_barrier(g);
            return result;
        }

        /**
         * @brief Reduce vector components with barriers between each reduction
         *
         * For AdaptiveCPP, we must reduce each component separately with
         * explicit barriers between reductions. This prevents corruption of
         * the group's internal synchronization state.
         *
         * @tparam T Vector type
         * @tparam Op Reduction operation type (plus, minimum, maximum)
         * @param g SYCL work-group
         * @param v Vector value to reduce
         * @param op Reduction operation
         * @return T Reduced vector
         */
        template<class T, class Op>
        inline T reduce_vector_with_barriers(const sycl::group<1> &g, const T &v, Op op) {
            using Tscal              = shambase::VecComponent<T>;
            static constexpr u32 dim = shambase::VectorProperties<T>::dimension;
            using ScalarOp           = Op;

            if constexpr (dim == 1) {
                return reduce_component(g, v, ScalarOp{});
            } else if constexpr (dim == 2) {
                auto r0 = reduce_component(g, v[0], ScalarOp{});
                auto r1 = reduce_component(g, v[1], ScalarOp{});
                return {r0, r1};
            } else if constexpr (dim == 3) {
                auto r0 = reduce_component(g, v[0], ScalarOp{});
                auto r1 = reduce_component(g, v[1], ScalarOp{});
                auto r2 = reduce_component(g, v[2], ScalarOp{});
                return {r0, r1, r2};
            } else if constexpr (dim == 4) {
                auto r0 = reduce_component(g, v[0], ScalarOp{});
                auto r1 = reduce_component(g, v[1], ScalarOp{});
                auto r2 = reduce_component(g, v[2], ScalarOp{});
                auto r3 = reduce_component(g, v[3], ScalarOp{});
                return {r0, r1, r2, r3};
            } else if constexpr (dim == 8) {
                auto r0 = reduce_component(g, v[0], ScalarOp{});
                auto r1 = reduce_component(g, v[1], ScalarOp{});
                auto r2 = reduce_component(g, v[2], ScalarOp{});
                auto r3 = reduce_component(g, v[3], ScalarOp{});
                auto r4 = reduce_component(g, v[4], ScalarOp{});
                auto r5 = reduce_component(g, v[5], ScalarOp{});
                auto r6 = reduce_component(g, v[6], ScalarOp{});
                auto r7 = reduce_component(g, v[7], ScalarOp{});
                return {r0, r1, r2, r3, r4, r5, r6, r7};
            } else if constexpr (dim == 16) {
                auto r0  = reduce_component(g, v[0], ScalarOp{});
                auto r1  = reduce_component(g, v[1], ScalarOp{});
                auto r2  = reduce_component(g, v[2], ScalarOp{});
                auto r3  = reduce_component(g, v[3], ScalarOp{});
                auto r4  = reduce_component(g, v[4], ScalarOp{});
                auto r5  = reduce_component(g, v[5], ScalarOp{});
                auto r6  = reduce_component(g, v[6], ScalarOp{});
                auto r7  = reduce_component(g, v[7], ScalarOp{});
                auto r8  = reduce_component(g, v[8], ScalarOp{});
                auto r9  = reduce_component(g, v[9], ScalarOp{});
                auto r10 = reduce_component(g, v[10], ScalarOp{});
                auto r11 = reduce_component(g, v[11], ScalarOp{});
                auto r12 = reduce_component(g, v[12], ScalarOp{});
                auto r13 = reduce_component(g, v[13], ScalarOp{});
                auto r14 = reduce_component(g, v[14], ScalarOp{});
                auto r15 = reduce_component(g, v[15], ScalarOp{});
                return {r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15};
            } else {
                static_assert(shambase::always_false_v<decltype(dim)>, "non-exhaustive visitor!");
            }
        }

    } // namespace detail

    /**
     * @brief Sum reduction across work-group
     *
     * Performs a parallel sum reduction of a value across all work-items
     * in the group. For vector types on AdaptiveCPP, uses explicit barriers
     * between component reductions to ensure correct synchronization.
     *
     * @tparam T Value type to reduce (scalar or vector)
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
        using Tscal = shambase::VecComponent<T>;
        return detail::reduce_vector_with_barriers(g, v, sycl::plus<Tscal>{});
    #endif
    }

    /**
     * @brief Minimum reduction across work-group
     *
     * Performs a parallel minimum reduction of a value across all work-items
     * in the group. For vector types on AdaptiveCPP, uses explicit barriers
     * between component reductions to ensure correct synchronization.
     *
     * @tparam T Value type to reduce (scalar or vector)
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
        using Tscal = shambase::VecComponent<T>;
        return detail::reduce_vector_with_barriers(g, v, sycl::minimum<Tscal>{});
    #endif
    }

    /**
     * @brief Maximum reduction across work-group
     *
     * Performs a parallel maximum reduction of a value across all work-items
     * in the group. For vector types on AdaptiveCPP, uses explicit barriers
     * between component reductions to ensure correct synchronization.
     *
     * @tparam T Value type to reduce (scalar or vector)
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
        using Tscal = shambase::VecComponent<T>;
        return detail::reduce_vector_with_barriers(g, v, sycl::maximum<Tscal>{});
    #endif
    }
#endif

} // namespace sham
