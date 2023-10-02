// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

namespace shamunits::details {

    template<int power,class T>
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
}