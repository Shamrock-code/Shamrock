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
 * @file typeAliasVec.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "typeAliasBase.hpp"
#include "typeAliasFp16.hpp"

#define TYPEDEFS_TYPES(...)                                                                        \
    using i64_##__VA_ARGS__ = sycl::vec<i64, __VA_ARGS__>;                                         \
    using i32_##__VA_ARGS__ = sycl::vec<i32, __VA_ARGS__>;                                         \
    using i16_##__VA_ARGS__ = sycl::vec<i16, __VA_ARGS__>;                                         \
    using i8_##__VA_ARGS__  = sycl::vec<i8, __VA_ARGS__>;                                          \
    using u64_##__VA_ARGS__ = sycl::vec<u64, __VA_ARGS__>;                                         \
    using u32_##__VA_ARGS__ = sycl::vec<u32, __VA_ARGS__>;                                         \
    using u16_##__VA_ARGS__ = sycl::vec<u16, __VA_ARGS__>;                                         \
    using u8_##__VA_ARGS__  = sycl::vec<u8, __VA_ARGS__>;                                          \
    using f16_##__VA_ARGS__ = sycl::vec<f16, __VA_ARGS__>;                                         \
    using f32_##__VA_ARGS__ = sycl::vec<f32, __VA_ARGS__>;                                         \
    using f64_##__VA_ARGS__ = sycl::vec<f64, __VA_ARGS__>;

TYPEDEFS_TYPES(2)
TYPEDEFS_TYPES(3)
TYPEDEFS_TYPES(4)
TYPEDEFS_TYPES(8)
TYPEDEFS_TYPES(16)

#undef TYPEDEFS_TYPES
