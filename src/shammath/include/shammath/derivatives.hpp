// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file derivatives.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief 
 * 
 */
 
#include <functional>

namespace shammath {

    template<class T>
    inline T derivative_upwind(T x, T dx, std::function<T(T)> &&fct) {
        return (fct(x +dx)-fct(x))/dx ;
    }

} // namespace shammath