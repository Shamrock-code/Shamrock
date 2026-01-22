// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file integrator.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <functional>

namespace shammath {

    template<class T, class Lambda>
    inline constexpr T integ_riemann_sum(T start, T end, T step, Lambda &&fct) {
        T acc = {};

        for (T x = start; x < end; x += step) {
            acc += fct(x) * step;
        }
        return acc;
    }

    /**
     * @brief Euler solving of ODE
     * The ode has the form
     * \f{eqnarray*}{
     * u'(x) &=& f(u,x) \\
     * u(x_0) &=& u_0
     * \f}
     * and will be solved between start and end with step $\mathrm{d}t$.
     *
     * @param start Lower bound of integration
     * @param end   Higher bound of integration
     * @param step  Step of integration $\mathrm{d}t$
     * @param ode   Ode function $f$
     * @param x0    Initial coordinate $x_0$
     * @param u0    Initial value $u_0$
     */
    template<class T, class Lambda>
    inline constexpr std::pair<std::vector<T>, std::vector<T>> euler_ode(
        T start, T end, T step, Lambda &&ode, T x0, T u0) {
        std::vector<T> U = {u0};
        std::vector<T> X = {x0};

        T u_prev = u0;
        T u      = u0;
        for (T x = x0 + step; x < end; x += step) {
            u = u_prev + ode(u_prev, x) * step;
            X.push_back(x);
            U.push_back(u);
            u_prev = u;
        };
        u_prev = u0;
        for (T x = x0 - step; x > start; x -= step) {
            u = u_prev - ode(u_prev, x) * step;
            X.insert(X.begin(), x);
            U.insert(U.begin(), u);
            u_prev = u;
        }
        return {X, U};
    }

} // namespace shammath
