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

#include <cstdint>

using i64 = std::int64_t;
using i32 = std::int32_t;
using i16 = std::int16_t;
using i8  = std::int8_t;
using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8  = std::uint8_t;

using byte = char;


constexpr u8 operator""_u8(unsigned long long n) { return u8(n); }
constexpr u16 operator""_u16(unsigned long long n) { return u16(n); }
constexpr u32 operator""_u32(unsigned long long n) { return u32(n); }
constexpr u64 operator""_u64(unsigned long long n) { return u64{n}; }
constexpr i8 operator""_i8(unsigned long long n) { return i8(n); }
constexpr i16 operator""_i16(unsigned long long n) { return i16(n); }
constexpr i32 operator""_i32(unsigned long long n) { return i32(n); }
constexpr i64 operator""_i64(unsigned long long n) { return i64(n); }
