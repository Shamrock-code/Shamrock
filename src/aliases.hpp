/**
 * @file aliases.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief typedefs and macros
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 * 
 */


#pragma once

#include <string.h>
#include <CL/sycl.hpp>


#define __FILENAME__ std::string(strstr(__FILE__, "/src/") ? strstr(__FILE__, "/src/")+1  : __FILE__)
#define throw_with_pos(...) throw std::runtime_error( __VA_ARGS__ " ("+ __FILENAME__ +":" + std::to_string(__LINE__) +")");

//#define PTR_FREE(...)      {if(__VA_ARGS__ != NULL){ delete   __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free \"" #__VA_ARGS__ "\" but it was already free'd");}}
//#define PTR_FREE_ARR(...)  {if(__VA_ARGS__ != NULL){ delete[] __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free array \"" #__VA_ARGS__ "\" but it was already free'd");}}

typedef cl::sycl::cl_long   i64;
typedef cl::sycl::cl_int    i32;
typedef cl::sycl::cl_short  i16;
typedef cl::sycl::cl_char   i8;
typedef cl::sycl::cl_ulong  u64;
typedef cl::sycl::cl_uint   u32;
typedef cl::sycl::cl_ushort u16;
typedef cl::sycl::cl_uchar  u8;
typedef cl::sycl::cl_half   f16;
typedef cl::sycl::cl_float  f32;
typedef cl::sycl::cl_double f64;

#define TYPEDEFS_TYPES(...) \
typedef cl::sycl::cl_long##__VA_ARGS__   i64_##__VA_ARGS__;\
typedef cl::sycl::cl_int##__VA_ARGS__    i32_##__VA_ARGS__;\
typedef cl::sycl::cl_short##__VA_ARGS__  i16_##__VA_ARGS__;\
typedef cl::sycl::cl_char##__VA_ARGS__   i8_##__VA_ARGS__;\
typedef cl::sycl::cl_ulong##__VA_ARGS__  u64_##__VA_ARGS__;\
typedef cl::sycl::cl_uint##__VA_ARGS__   u32_##__VA_ARGS__;\
typedef cl::sycl::cl_ushort##__VA_ARGS__ u16_##__VA_ARGS__;\
typedef cl::sycl::cl_uchar##__VA_ARGS__  u8_##__VA_ARGS__;\
typedef cl::sycl::cl_half##__VA_ARGS__   f16_##__VA_ARGS__;\
typedef cl::sycl::cl_float##__VA_ARGS__  f32_##__VA_ARGS__;\
typedef cl::sycl::cl_double##__VA_ARGS__ f64_##__VA_ARGS__;\

TYPEDEFS_TYPES(2)
TYPEDEFS_TYPES(3)
TYPEDEFS_TYPES(4)
TYPEDEFS_TYPES(8)
TYPEDEFS_TYPES(16)

#define ERR_ID_64 18446744073709551615u


extern std::string git_info_str;

inline std::string shamrock_title_bar_big = "\n\
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████\n\
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░ \n\
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███   \n\
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████    \n\
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███   \n\
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███  \n\
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████\n\
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░ \n\
";

inline void print_title_bar(){
    printf("%s\n",shamrock_title_bar_big.c_str());
    printf("---------------------------------------------------------------------------------");
    printf("%s\n",git_info_str.c_str());
    printf("---------------------------------------------------------------------------------\n");

}