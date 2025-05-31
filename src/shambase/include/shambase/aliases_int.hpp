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
 * @file aliases_int.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <cstddef>
#include <cstdint>

using i64   = std::int64_t;  ///< 64 bit integer
using i32   = std::int32_t;  ///< 32 bit integer
using i16   = std::int16_t;  ///< 16 bit integer
using i8    = std::int8_t;   ///< 8 bit integer
using u64   = std::uint64_t; ///< 64 bit unsigned integer
using u32   = std::uint32_t; ///< 32 bit unsigned integer
using u16   = std::uint16_t; ///< 16 bit unsigned integer
using u8    = std::uint8_t;  ///< 8 bit unsigned integer
using usize = std::size_t;   ///< size_t alias

using byte = char; ///< byte type similar to std::byte

/// Literal suffixes for integer types
///
/// These allow for a more convenient and less error-prone way of specifying
/// integer literals of different types.
///
/// Example:
///     42_u8  // of type u8
///     0x1_i32 // of type i32

/// Literal suffix for u8
constexpr u8 operator""_u8(unsigned long long n) { return u8(n); }
/// Literal suffix for u16
constexpr u16 operator""_u16(unsigned long long n) { return u16(n); }
/// Literal suffix for u32
constexpr u32 operator""_u32(unsigned long long n) { return u32(n); }
/// Literal suffix for u64
constexpr u64 operator""_u64(unsigned long long n) { return u64{n}; }
/// Literal suffix for i8
constexpr i8 operator""_i8(unsigned long long n) { return i8(n); }
/// Literal suffix for i16
constexpr i16 operator""_i16(unsigned long long n) { return i16(n); }
/// Literal suffix for i32
constexpr i32 operator""_i32(unsigned long long n) { return i32(n); }
/// Literal suffix for i64
constexpr i64 operator""_i64(unsigned long long n) { return i64(n); }
