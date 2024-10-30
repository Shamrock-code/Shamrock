// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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
        return (fct(x + dx) - fct(x)) / dx;
    }

} // namespace shammath
