// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file integrator.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include <functional>

namespace shammath {

    template<class T>
    inline T integ_riemann_sum(T start, T end, T step, std::function<T(T)> &&fct) {
        T acc = {};

        for (T x = start; x < end; x += step) {
            acc += fct(x) * step;
        }
        return acc;
    }

} // namespace shammath