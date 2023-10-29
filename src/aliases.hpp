// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file aliases.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief typedefs and macros
 * @date 2021-09-17
 * @copyright Copyright Timothée David--Cléris (c) 2021
 * 
 */


//%Impl status : Good

//#pragma message("This file is deprecated please use the alias include directly")


#include <cstring>
#include <string>
#include <type_traits>
#include "shambackends/typeAliasVec.hpp"

inline std::string __file_to_loc(const char* filename){
    return std::string(std::strstr(filename, "/src/") ? std::strstr(filename, "/src/")+1  : filename);
}

inline std::string __loc_prefix(const char* filename, int line){
    return __file_to_loc(filename)+":" + std::to_string(line);
}

#define __FILENAME__ __file_to_loc(__FILE__)
#define __LOC_PREFIX__  __loc_prefix(__FILE__,__LINE__)

#define __LOC_POSTFIX__  ("("+__LOC_PREFIX__+")")
//#define throw_with_pos(...) throw std::runtime_error( __VA_ARGS__ " ("+ __FILENAME__ +":" + std::to_string(__LINE__) +")");


//#define excep_with_pos(a, ...) a ((std::string(__VA_ARGS__) + "\n-------------------\n - at:\n    "+__LOC_PREFIX__ +"\n - call:\n    "+std::string(__PRETTY_FUNCTION__)+"\n-------------------").c_str())

//#define PTR_FREE(...)      {if(__VA_ARGS__ != NULL){ delete   __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free \"" #__VA_ARGS__ "\" but it was already free'd");}}
//#define PTR_FREE_ARR(...)  {if(__VA_ARGS__ != NULL){ delete[] __VA_ARGS__; __VA_ARGS__ = NULL; }else{ throw_with_pos("trying to free array \"" #__VA_ARGS__ "\" but it was already free'd");}}

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>;




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


