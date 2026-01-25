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
 * @file solve.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammath/matrix.hpp"
#include "shammath/matrix_op.hpp"
#include <cmath>
#include <functional>

namespace shammath {

    template<class T>
    float newton_rhaphson(std::function<T(T)> &&f, std::function<T(T)> &&df, T epsilon_c, T x_0) {

        auto iterate_newton = [](T f, T df, T xk) -> T {
            return xk - (f / df);
        };

        T xk      = x_0;
        T epsilon = 100000;

        while (epsilon > epsilon_c) {
            T xkp1 = iterate_newton(f(xk), df(xk), xk);

            epsilon = std::fabs(xk - xkp1);

            xk = xkp1;
        }

        return xk;
    }

    /**
     * @brief This function determines the best fit parameters $\vec p$ for a given function(f(\vec
     * (beta), \mathbf(X)) with least squares.
     *
     * @param f     Function (1d values)
     * @param X     $x$ Data to fit
     * @param Y     $y$ Data to fit
     * @param p0    Initial parameters guessed
     *
     * @details The Levenberg-Marquardt method is used. Therefore, the number of observations needs
     * to be greater than the number of paremeters.
     */
    template<class T, class Lambda>
    std::vector<T> least_squares(
        const Lambda &f,
        const std::vector<T> &X,
        const std::vector<T> &Y,
        const std::vector<T> &p0) {
        SHAM_ASSERT(X.size() == Y.size());

        const int params_nb = p0.size();
        const int data_size = X.size();

        std::vector<T> p = p0;
        T mu             = 1e-2; // damping parameter
        T beta           = 0.1;  // decay rate
        int maxits       = 1000;
        int it           = 0;
        T sse            = 0.0;
        for (int k = 0; k < X.size(); k++) {
            T r = Y[k] - f(p, X[k]);
            sse += r * r;
        };
        while (it < maxits) {

            // Construct the Jaobian (finite differences)
            shammath::mat_d<T> J(data_size, params_nb);
            mat_set_vals(J.get_mdspan(), [&](auto i, auto j) -> T {
                auto p_plus_dpj = p;
                T step_scale    = (std::abs(p_plus_dpj[j]) < 1e-6) ? 1e-6 : p_plus_dpj[j];
                T dpj           = step_scale * 0.001;
                p_plus_dpj[j] += dpj;
                return (f(p_plus_dpj, X[i]) - f(p, X[i])) / dpj;
            });

            shammath::vec_d<T> R(data_size);
            shammath::vec_set_vals(R.get_mdspan(), [&](auto i) -> T {
                return Y[i] - f(p, X[i]);
            });

            shammath::mat_d<T> J_T(params_nb, data_size); // Jacobian tranpose
            shammath::mat_transpose(J.get_mdspan(), J_T.get_mdspan());

            shammath::mat_d<T> G(params_nb, params_nb); // left hand side
            shammath::mat_prod(J_T.get_mdspan(), J.get_mdspan(), G.get_mdspan());
            shammath::mat_plus_equal_scalar_id(G.get_mdspan(), mu);

            shammath::vec_d<T> d(params_nb); // right hand side
            shammath::mat_gemv(1.0, J_T.get_mdspan(), R.get_mdspan(), 0.0, d.get_mdspan());

            shammath::vec_d<T> delta(params_nb); // increment for p
            shammath::Cholesky_solve(G.get_mdspan(), d.get_mdspan(), delta.get_mdspan());

            std::vector<T> p_trial = p;
            for (int i = 0; i < params_nb; i++) {
                p_trial[i] += delta.data[i];
            };
            T sse_trial = 0.0;
            for (int k = 0; k < X.size(); k++) {
                sse_trial += (Y[k] - f(p_trial, X[k])) * (Y[k] - f(p_trial, X[k]));
            };
            if (sse_trial > sse) { // Fail -> gradient descent
                mu /= beta;
            } else { // Not bad -> Gauss-Newton
                // it++;
                mu *= beta;
                p = p_trial;
            }
            it++;
        };

        return p;
    }
} // namespace shammath
