// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file aliases_float.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

using f32 = float;  ///< Alias for float
using f64 = double; ///< Alias for double

/// Literal suffix for 32 bit float
constexpr f32 operator""_f32(long double n) { return f32(n); }
/// Literal suffix for 64 bit float
constexpr f64 operator""_f64(long double n) { return f64(n); }
