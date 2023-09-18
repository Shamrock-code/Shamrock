// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include <climits>
#include "shambase/type_traits.hpp"

namespace shambase {

    /**
     * @brief determine if v is a power of two
     * Warning : this function return true if v == 0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr bool is_pow_of_two_fast(T v) noexcept{
        return (v & (v - 1)) == 0;
    }

    /**
     * @brief determine if v is a power of two and check if v==0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr bool is_pow_of_two(T v) noexcept{
        return v && !(v & (v - 1));
    }

    /**
     * @brief check if the sign of the two integers differs
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T
     * @param a
     * @param b
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || std::is_signed_v<T>, int> = 0>
    inline constexpr bool sign_differ(T a, T b) noexcept {
        return ((a ^ b) < 0);
    }

    /**
     * @brief swap two values using xor
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * 
     * @tparam T 
     * @param a 
     * @param b 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline void xor_swap(T &a, T &b) {
        a ^= b;
        b ^= a;
        a ^= b;
    };

    /**
     * @brief round up to the next power of two
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * 
     * @tparam T 
     * @param v 
     * @return constexpr T 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2 (T v) noexcept {
        
        if constexpr (has_bitlen_v<T, 32>){
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return v;
        }

        if constexpr (has_bitlen_v<T, 64>){
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v++;
            return v;
        }
    };



    inline constexpr u32 group_count(u32 len, u32 group_size){
        return (len+group_size-1)/group_size;
    }

    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline T select_bit(T value, T bitnum){
        return (value >> bitnum) & 1;
    }


    inline auto pack(u32 a, u32 b) -> u64 {
        return (u64(a) << 32U) + b;
    };

    inline auto unpack (u64 v) -> sycl::vec<u32, 2> {
        return {u32(v >> 32U), u32(v)};
    };


    
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0> 
    inline constexpr T most_sig_bit_mask() noexcept {
        return T(std::make_unsigned_t<T>(1) << (sizeof(T) * 8 - 1));
    }

    
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0> 
    inline constexpr bool is_most_sig_bit_set(const T x)noexcept{
        return (x & most_sig_bit_mask<T>());
    }


    template<i32 power,class T>
    inline constexpr T pow_constexpr(T a) noexcept {

        if constexpr (power == 0){
            return T{1};
        }else if constexpr (power % 2 == 0){
            T tmp = pow_constexpr<power/2>(a);
            return tmp*tmp;
        }else if constexpr (power % 2 == 1){
            T tmp = pow_constexpr<(power-1)/2>(a);
            return tmp*tmp*a;
        }

    }


} // namespace shambase