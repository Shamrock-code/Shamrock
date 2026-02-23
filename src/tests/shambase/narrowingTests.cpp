// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/narrowing.hpp"
#include "shamtest/shamtest.hpp"

// Compile-time tests
namespace {
    // Same type conversions - should always succeed
    static_assert(shambase::can_narrow<i32>(42_i32), "Same type conversion should succeed");
    static_assert(shambase::can_narrow<u32>(42_u32), "Same type conversion should succeed");

    // Widening conversions - should succeed
    static_assert(shambase::can_narrow<i32>(100_i16), "Widening conversion should succeed");
    static_assert(shambase::can_narrow<i64>(42_i32), "Widening conversion should succeed");
    static_assert(shambase::can_narrow<u64>(42_u32), "Widening conversion should succeed");

    // Narrowing conversions that fit - should succeed
    static_assert(shambase::can_narrow<i8>(100_i32), "Value 100 fits in i8");
    static_assert(shambase::can_narrow<i16>(100_i32), "Value 100 fits in i16");
    static_assert(shambase::can_narrow<u8>(200_u32), "Value 200 fits in u8");
    static_assert(shambase::can_narrow<i8>(127_i32), "Value 127 fits in i8");
    static_assert(shambase::can_narrow<i8>(-128_i32), "Value -128 fits in i8");

    // Narrowing conversions that don't fit - should fail
    static_assert(!shambase::can_narrow<i8>(300_i32), "Value 300 does not fit in i8");
    static_assert(!shambase::can_narrow<i8>(128_i32), "Value 128 does not fit in i8");
    static_assert(!shambase::can_narrow<i8>(-129_i32), "Value -129 does not fit in i8");
    static_assert(!shambase::can_narrow<u8>(256_u32), "Value 256 does not fit in u8");
    static_assert(!shambase::can_narrow<i16>(100000_i32), "Value 100000 does not fit in i16");
    static_assert(
        !shambase::can_narrow<u32>(5000000000_u64), "Value 5000000000 does not fit in u32");

    // Signed to unsigned conversions
    static_assert(shambase::can_narrow<u32>(42_i32), "Positive value converts to unsigned");
    static_assert(
        !shambase::can_narrow<u32>(-1_i32), "Negative value does not convert to unsigned");
    static_assert(
        !shambase::can_narrow<u8>(-10_i32), "Negative value does not convert to unsigned");

    // Unsigned to signed conversions
    static_assert(shambase::can_narrow<i32>(42_u32), "Small unsigned converts to signed");
    static_assert(
        !shambase::can_narrow<i32>(0xFFFFFFFF_u32), "Large unsigned does not fit in signed");
    static_assert(!shambase::can_narrow<i8>(200_u32), "Value 200u does not fit in i8");
    static_assert(shambase::can_narrow<i8>(100_u32), "Value 100u fits in i8");

    // Edge cases at type boundaries
    static_assert(shambase::can_narrow<i8>(127_i32), "Max i8");
    static_assert(shambase::can_narrow<i8>(-128_i32), "Min i8");
    static_assert(!shambase::can_narrow<i8>(128_i32), "Max i8 + 1");
    static_assert(!shambase::can_narrow<i8>(-129_i32), "Min i8 - 1");
    static_assert(shambase::can_narrow<u8>(255_u32), "Max u8");
    static_assert(shambase::can_narrow<u8>(0_u32), "Min u8");
    static_assert(!shambase::can_narrow<u8>(256_u32), "Max u8 + 1");
} // namespace
