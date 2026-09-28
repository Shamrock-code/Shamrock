// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file flatten.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Attribute forcing the inlining of a function and of every call made from its body.
 *
 * Intended for small helpers that loop over a range and invoke a user provided functor at every
 * iteration (for example tree traversal primitives). The attribute is placed on the helper :
 *
 * @code{.cpp}
 * template<class Functor>
 * SHAM_ALWAYS_INLINE_FLATTEN inline void for_each_in_range(u32 begin, u32 end, Functor &&func) {
 *     for (u32 i = begin; i < end; i++) {
 *         func(i);
 *     }
 * }
 * @endcode
 *
 * so that at the call site both the helper and the functor are inlined into the caller's body :
 *
 * @code{.cpp}
 * Tscal sum = 0;
 * for_each_in_range(begin, end, [&](u32 i) {
 *     sum += values[i];
 * });
 * @endcode
 *
 * Without the attribute the compiler may keep `func` as an out of line call for every iteration,
 * in which case the captured variables (here `sum`) are read from and written to memory at every
 * call instead of staying in registers. Inlining only changes the generated code, not the
 * operations performed.
 *
 * `flatten` inlines the whole call tree of the function body, so avoid it on functions whose
 * body calls large or non trivial functions, as it can blow up the code size.
 *
 * Can be disabled by configuring with -DSHAMROCK_USE_ALWAYS_INLINE_FLATTEN=Off (defines
 * SHAMROCK_DISABLE_ALWAYS_INLINE_FLATTEN).
 */

#if !defined(SHAMROCK_DISABLE_ALWAYS_INLINE_FLATTEN) && (defined(__clang__) || defined(__GNUC__))
    /// Force the inlining of every call made from the function body
    #define SHAM_ALWAYS_INLINE_FLATTEN __attribute__((always_inline, flatten))
#else
    /// Force the inlining of every call made from the function body (disabled)
    #define SHAM_ALWAYS_INLINE_FLATTEN
#endif
