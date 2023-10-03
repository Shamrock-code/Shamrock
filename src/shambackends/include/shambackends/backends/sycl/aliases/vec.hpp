// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "basetypes.hpp"
#include "half.hpp"

namespace sham {
    template<typename T, int N>
    using vec = sycl::vec<T, N>;
} // namespace sham

#define TYPEDEFS_TYPES(...)                                                                        \
    using i64_##__VA_ARGS__ = sham::vec<i64, __VA_ARGS__>;                                         \
    using i32_##__VA_ARGS__ = sham::vec<i32, __VA_ARGS__>;                                         \
    using i16_##__VA_ARGS__ = sham::vec<i16, __VA_ARGS__>;                                         \
    using i8_##__VA_ARGS__  = sham::vec<i8, __VA_ARGS__>;                                          \
    using u64_##__VA_ARGS__ = sham::vec<u64, __VA_ARGS__>;                                         \
    using u32_##__VA_ARGS__ = sham::vec<u32, __VA_ARGS__>;                                         \
    using u16_##__VA_ARGS__ = sham::vec<u16, __VA_ARGS__>;                                         \
    using u8_##__VA_ARGS__  = sham::vec<u8, __VA_ARGS__>;                                          \
    using f16_##__VA_ARGS__ = sham::vec<f16, __VA_ARGS__>;                                         \
    using f32_##__VA_ARGS__ = sham::vec<f32, __VA_ARGS__>;                                         \
    using f64_##__VA_ARGS__ = sham::vec<f64, __VA_ARGS__>;

TYPEDEFS_TYPES(2)
TYPEDEFS_TYPES(3)
TYPEDEFS_TYPES(4)
TYPEDEFS_TYPES(8)
TYPEDEFS_TYPES(16)

#undef TYPEDEFS_TYPES