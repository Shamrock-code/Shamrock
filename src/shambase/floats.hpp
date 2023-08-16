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
#include "shambase/vectors.hpp"

namespace shambase {

    template<i32 power,class T>
    inline constexpr T pow_constexpr_fast_inv(T a, T a_inv) noexcept {

        if constexpr (power < 0){
            return pow_constexpr_fast_inv<-power>(a_inv,a);
        }else if constexpr (power == 0){
            return T{1};
        }else if constexpr (power % 2 == 0){
            T tmp = pow_constexpr_fast_inv<power/2>(a,a_inv);
            return tmp*tmp;
        }else if constexpr (power % 2 == 1){
            T tmp = pow_constexpr_fast_inv<(power-1)/2>(a,a_inv);
            return tmp*tmp*a;
        }

    }

    template<class T>
    inline bool has_nan(T v){
        auto tmp = ! sycl::isnan(v);
        return !(tmp);
    }

    template<class T>
    inline bool has_inf(T v){
        auto tmp = ! sycl::isinf(v);
        return !(tmp);
    }

    template<class T>
    inline bool has_nan_or_inf(T v){
        auto tmp = ! (sycl::isnan(v) || sycl::isinf(v));
        return !(tmp);
    }

    /**
     * @brief return true if vector has a nan
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_nan(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            has = has || (sycl::isnan(v[i]));
        }
        return has;
    }

    /**
     * @brief return true if vector has a inf
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_inf(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            has = has || (sycl::isinf(v[i]));
        }
        return has;
    }

    /**
     * @brief return true if vector has a nan or a inf
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_nan_or_inf(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            has = has || (sycl::isnan(v[i]) || sycl::isinf(v[i]));
        }
        return has;
    }


}