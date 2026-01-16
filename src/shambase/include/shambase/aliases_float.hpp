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
 * @file aliases_float.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

using f32 = float;  ///< Alias for float
using f64 = double; ///< Alias for double

/// Literal suffix for 32 bit float
constexpr f32 operator""_f32(long double n) { return f32(n); }
/// Literal suffix for 64 bit float
constexpr f64 operator""_f64(long double n) { return f64(n); }
