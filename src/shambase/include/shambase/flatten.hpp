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
 * Used on the small loop helpers taking a functor (e.g. neighbour loops) : without it the
 * compiler may keep the functor as an out of line call for every iteration, with its captured
 * accumulators living in memory instead of registers. Inlining does not change the floating
 * point operations performed, only the generated code.
 *
 * Can be disabled by configuring with -DSHAMROCK_FLATTEN_LOOPS=Off (defines
 * SHAMROCK_DISABLE_FLATTEN_LOOPS).
 */

#if !defined(SHAMROCK_DISABLE_FLATTEN_LOOPS) && (defined(__clang__) || defined(__GNUC__))
    /// Force the inlining of every call made from the function body
    #define SHAM_ALWAYS_INLINE_FLATTEN __attribute__((always_inline, flatten))
#else
    /// Force the inlining of every call made from the function body (disabled)
    #define SHAM_ALWAYS_INLINE_FLATTEN
#endif
