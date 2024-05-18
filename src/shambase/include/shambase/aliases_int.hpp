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

using i64 = std::int64_t;
using i32 = std::int32_t;
using i16 = std::int16_t;
using i8  = std::int8_t;
using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8  = std::uint8_t;
using usize = std::size_t;

using byte = char;


constexpr u8 operator""_u8(unsigned long long n) { return u8(n); }
constexpr u16 operator""_u16(unsigned long long n) { return u16(n); }
constexpr u32 operator""_u32(unsigned long long n) { return u32(n); }
constexpr u64 operator""_u64(unsigned long long n) { return u64{n}; }
constexpr i8 operator""_i8(unsigned long long n) { return i8(n); }
constexpr i16 operator""_i16(unsigned long long n) { return i16(n); }
constexpr i32 operator""_i32(unsigned long long n) { return i32(n); }
constexpr i64 operator""_i64(unsigned long long n) { return i64(n); }

#ifndef INT_ALIAS_LIM_DEFINED

constexpr i64 i64_max = 0x7FFFFFFFFFFFFFFF;
constexpr i32 i32_max = 0x7FFFFFFF;
constexpr i16 i16_max = 0x7FFF;
constexpr i8  i8_max  = 0x7F;

constexpr i64 i64_min = 0x8000000000000000;
constexpr i32 i32_min = 0x80000000;
constexpr i16 i16_min = 0x8000;
constexpr i8  i8_min  = 0x80;

constexpr u64 u64_max = 0xFFFFFFFFFFFFFFFF;
constexpr u32 u32_max = 0xFFFFFFFF;
constexpr u16 u16_max = 0xFFFF;
constexpr u8  u8_max  = 0xFF;

constexpr u64 u64_min = 0x0000000000000000;
constexpr u32 u32_min = 0x00000000;
constexpr u16 u16_min = 0x0000;
constexpr u8  u8_min  = 0x00;

#endif