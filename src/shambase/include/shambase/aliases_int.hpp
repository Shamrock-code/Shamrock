// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file aliases_int.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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

#ifndef INT_ALIAS_LIM_DEFINED

constexpr i64 i64_max = 0x7FFFFFFFFFFFFFFF; ///< i64 max value
constexpr i32 i32_max = 0x7FFFFFFF;         ///< i32 max value
constexpr i16 i16_max = 0x7FFF;             ///< i16 max value
constexpr i8 i8_max   = 0x7F;               ///< i8 max value

constexpr i64 i64_min = 0x8000000000000000; ///< i64 min value
constexpr i32 i32_min = 0x80000000;         ///< i32 min value
constexpr i16 i16_min = 0x8000;             ///< i16 min value
constexpr i8 i8_min   = 0x80;               ///< i8 min value

constexpr u64 u64_max = 0xFFFFFFFFFFFFFFFF; ///< u64 max value
constexpr u32 u32_max = 0xFFFFFFFF;         ///< u32 max value
constexpr u16 u16_max = 0xFFFF;             ///< u16 max value
constexpr u8 u8_max   = 0xFF;               ///< u8 max value

constexpr u64 u64_min = 0x0000000000000000; ///< u64 min value
constexpr u32 u32_min = 0x00000000;         ///< u32 min value
constexpr u16 u16_min = 0x0000;             ///< u16 min value
constexpr u8 u8_min   = 0x00;               ///< u8 min value

#endif
