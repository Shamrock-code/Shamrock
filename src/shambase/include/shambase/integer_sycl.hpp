// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_aliases.hpp"
#include "shambase/type_traits.hpp"
#include "shambase/sycl.hpp"

namespace shambase {

    

    #ifdef SYCL_COMP_DPCPP

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{
        return sycl::clz(a);
    }

    #endif

    #ifdef SYCL_COMP_OPENSYCL

    namespace details{
        template<class T>
        int internal_clz(T a);

        template<>
        inline int internal_clz(u32 a){
            return __builtin_clz(a);
        }

        template<>
        inline int internal_clz(u64 a){
            return __builtin_clzl(a);
        }
    }

    template<class T>
    inline constexpr T sycl_clz(T a) noexcept{

        __hipsycl_if_target_host(
            return details::internal_clz(a);
        )

        __hipsycl_if_target_hiplike(
            return __clz(a);
        )

        __hipsycl_if_target_spirv(
            return __clz(a);
        )

        return 0; //weird fixes waiting for https://github.com/OpenSYCL/OpenSYCL/pull/965

    }

    #endif

    /**
     * @brief give the lenght of the common prefix
     *
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr T clz_xor(T a, T b) noexcept{
        return sycl_clz(a^b);
    }

    /**
     * @brief round up to the next power of two
     * CLZ version
     * 
     * @tparam T 
     * @param v 
     * @return constexpr T 
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2_clz (T v) noexcept {

        T clz_val = shambase::sycl_clz(v);

        T val_rounded_pow = 1U << (bitsizeof<T>-clz_val);
        if(v == 1U << (bitsizeof<T>-clz_val-1)){
            val_rounded_pow = v;
        }

        return val_rounded_pow; 
    };

    /**
     * @brief delta operator defined in Karras 2012
     * 
     * @tparam Acc 
     * @param x 
     * @param y 
     * @param morton_lenght 
     * @param m 
     * @return i32 
     */
    template<class Acc>
    inline i32 karras_delta(i32 x, i32 y, u32 morton_lenght, Acc m) noexcept {
        return ((y > morton_lenght - 1 || y < 0) ? -1 : int(clz_xor(m[x] , m[y])));
    }

}