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
 * @file unique_name_macro.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Provides macros for generating unique identifiers at compile time.
 *
 */

/// Utility to concatenate two tokens
#define internal_macro_sham_CONCAT2(a, b) a##b
/// Utility to expand a macro with two tokens
#define internal_macro_sham_EXPAND2(a, b) internal_macro_sham_CONCAT2(a, b)

/**
 * @fn __shamrock_unique_name
 * @brief Macro to create a unique name.
 *
 * This macro creates a unique identifier from `base_name` using `__COUNTER__` or `__LINE__`.
 * @note The `__LINE__` fallback is not unique for multiple uses on the same line.
 */

#ifdef __COUNTER__
    #define __shamrock_unique_name(base_name) internal_macro_sham_EXPAND2(base_name, __COUNTER__)
#else
    #define __shamrock_unique_name(base_name) internal_macro_sham_EXPAND2(base_name, __LINE__)
#endif
