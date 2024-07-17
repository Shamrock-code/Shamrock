// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file solve.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include <functional>

namespace shammath {

    template<class T>
    float newtown_rhaphson(std::function<T(T)> &&f, std::function<T(T)> &&df, T epsilon_c, T x_0) {

        auto iterate_newtown = [](T f, T df, T xk) -> T {
            return xk - (f / df);
        };

        T xk      = x_0;
        T epsilon = 100000;

        while (epsilon > epsilon_c) {
            T xkp1 = iterate_newtown(f(xk), df(xk), xk);

            epsilon = std::abs(xk - xkp1);

            xk = xkp1;
        }

        return xk;
    }

} // namespace shammath
