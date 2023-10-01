// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

//#include <sycl/sycl.hpp>

#ifdef SYCL_COMP_ACPP
#include <hipSYCL/sycl/libkernel/vec.hpp>
#include <hipSYCL/sycl/types.hpp>

//copied from hipsycl sycl/sycl.hpp
namespace sycl {
    using namespace hipsycl::sycl;
}

using i64 = sycl::detail::s_long  ;
using i32 = sycl::detail::s_int   ;
using i16 = sycl::detail::s_short ;
using i8  = sycl::detail::s_char  ;
using u64 = sycl::detail::u_long  ;
using u32 = sycl::detail::u_int   ;
using u16 = sycl::detail::u_short ;
using u8  = sycl::detail::u_char  ;
using f16 = u16; // issue with hipsycl not supporting half
using f32 = sycl::detail::sp_float;
using f64 = sycl::detail::dp_float;

#endif

#ifdef SYCL_COMP_INTEL_LLVM
#include <cstdint>
#include <detail/generic_type_lists.hpp>
#include <sycl/types.hpp>
using i64 = std::int64_t  ;
using i32 = std::int32_t  ;
using i16 = std::int16_t  ;
using i8  = std::int8_t   ;
using u64 = std::uint64_t ;
using u32 = std::uint32_t ;
using u16 = std::uint16_t ;
using u8  = std::uint8_t  ;
using f16 = sycl::half    ;
using f32 = float         ;
using f64 = double        ;

#endif


constexpr u8  operator""_u8 (unsigned long long n){return u8(n);}
constexpr u16 operator""_u16(unsigned long long n){return u16(n);}
constexpr u32 operator""_u32(unsigned long long n){return u32(n);}
constexpr u64 operator""_u64(unsigned long long n){return u64{n};}
constexpr i8  operator""_i8 (unsigned long long n){return i8(n);}
constexpr i16 operator""_i16(unsigned long long n){return i16(n);}
constexpr i32 operator""_i32(unsigned long long n){return i32(n);}
constexpr i64 operator""_i64(unsigned long long n){return i64(n);}
constexpr f16 operator""_f16(long double n){return f16(n);}
constexpr f32 operator""_f32(long double n){return f32(n);}
constexpr f64 operator""_f64(long double n){return f64(n);}



#define TYPEDEFS_TYPES(...) \
using i64_##__VA_ARGS__ = sycl::vec<i64,__VA_ARGS__>;\
using i32_##__VA_ARGS__ = sycl::vec<i32,__VA_ARGS__>;\
using i16_##__VA_ARGS__ = sycl::vec<i16,__VA_ARGS__>;\
using i8_##__VA_ARGS__  = sycl::vec<i8 ,__VA_ARGS__>;\
using u64_##__VA_ARGS__ = sycl::vec<u64,__VA_ARGS__>;\
using u32_##__VA_ARGS__ = sycl::vec<u32,__VA_ARGS__>;\
using u16_##__VA_ARGS__ = sycl::vec<u16,__VA_ARGS__>;\
using u8_##__VA_ARGS__  = sycl::vec<u8 ,__VA_ARGS__>;\
using f16_##__VA_ARGS__ = sycl::vec<f16,__VA_ARGS__>;\
using f32_##__VA_ARGS__ = sycl::vec<f32,__VA_ARGS__>;\
using f64_##__VA_ARGS__ = sycl::vec<f64,__VA_ARGS__>;\

TYPEDEFS_TYPES(2)
TYPEDEFS_TYPES(3)
TYPEDEFS_TYPES(4)
TYPEDEFS_TYPES(8)
TYPEDEFS_TYPES(16)

#undef TYPEDEFS_TYPES