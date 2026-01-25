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
    std::tuple<std::vector<T>, T> least_squares(
        const Lambda &f,
        const std::vector<T> &X,
        const std::vector<T> &Y,
        const std::vector<T> &p0,
        int maxits  = 1000,
        T tolerence = 1e-6) {
        SHAM_ASSERT(X.size() == Y.size());

        const int params_nb = p0.size();
        const int data_size = X.size();

        std::vector<T> p = p0;
        T mu             = 1e-2; // damping parameter
        T beta           = 0.1;  // decay rate
        int it           = 0;
        T sse            = 0.0;
        for (int k = 0; k < X.size(); k++) {
            T r = Y[k] - f(p, X[k]);
            sse += r * r;
        };
        T sse_trial = 999.0;
        while (it < maxits and sham::abs(sse_trial - sse) > tolerence) {
            sse = 0.0;
            for (int k = 0; k < X.size(); k++) {
                T r = Y[k] - f(p, X[k]);
                sse += r * r;
            };

            // Construct the Jacobian (finite differences)
            shammath::mat_d<T> J(data_size, params_nb);
            std::vector<T> f_at_p(data_size);
            for (int i = 0; i < data_size; i++) {
                f_at_p[i] = f(p, X[i]);
            }
            mat_set_vals(J.get_mdspan(), [&](auto i, auto j) -> T {
                T original_p_j = p[j];
                T step_scale   = (std::abs(original_p_j) < 1e-6) ? 1e-6 : original_p_j;
                T dpj          = step_scale * 0.001;

                p[j] += dpj;
                T f_perturbed = f(p, X[i]);
                p[j]          = original_p_j; // Restore

                return (f_perturbed - f_at_p[i]) / dpj;
            });

            shammath::vec_d<T> R(data_size);
            shammath::vec_set_vals(R.get_mdspan(), [&](auto i) -> T {
                return Y[i] - f(p, X[i]);
            });

            shammath::mat_d<T> J_T(params_nb, data_size); // Jacobian transposed
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

            sse_trial = 0.0;
            for (int k = 0; k < X.size(); k++) {
                T residual = Y[k] - f(p_trial, X[k]);
                sse_trial += residual * residual;
            };
            if (sse_trial > sse) { // Fail -> gradient descent
                mu /= beta;
            } else { // Not bad -> Gauss-Newton
                it++;
                mu *= beta;
                p = p_trial;
            }
        };

        T total_sum_squares = 0.0;
        T mean_Y            = 0.0;
        for (int k = 0; k < Y.size(); k++) {
            mean_Y += Y[k];
        }
        mean_Y /= Y.size();
        for (int k = 0; k < Y.size(); k++) {
            total_sum_squares += (Y[k] - mean_Y) * (Y[k] - mean_Y);
        }
        T R2 = 1 - sse / total_sum_squares;

        shamlog_debug_ln(
            "least_squares", "Least squares stopped after", it, "iterations with R^2=", R2);
        return {p, R2};
    }
} // namespace shammath
