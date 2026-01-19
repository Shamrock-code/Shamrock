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
 * @file derivatives.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambackends/sycl.hpp"
#include <functional>

namespace shammath {

    /**
     * @brief Compute the derivative of a function at x using the upwind method.
     *
     * The upwind method is a first order method that uses the current value of the function
     * and the value at the neighboring point to compute the derivative. The neighboring
     * point is chosen such that it is located in the direction of the flow.
     *
     * @param x point at which to evaluate the derivative
     * @param dx spacing between two consecutive points
     * @param fct function to evaluate
     * @return the derivative of the function at x
     */
    template<class T>
    inline T derivative_upwind(T x, T dx, std::function<T(T)> &&fct) {
        return (fct(x + dx) - fct(x)) / dx;
    }

    /**
     * @brief Compute the derivative of a function at x using the centered difference method.
     *
     * The centered difference method computes the derivative using the average rate of change
     * between the points x + dx and x - dx. This method is second-order accurate.
     *
     * @param x point at which to evaluate the derivative
     * @param dx spacing between two consecutive points
     * @param fct function to evaluate
     * @return the derivative of the function at x
     */

    template<class T>
    inline T derivative_centered(T x, T dx, std::function<T(T)> &&fct) {
        return (fct(x + dx) - fct(x - dx)) / (2 * dx);
    }

    /**
     * @brief Compute the derivative of a function at x using a 3-point forward finite difference.
     *
     * This method computes the derivative using the points x, x + dx and x + 2*dx. This method
     * has a second-order accuracy.
     *
     * @param x point at which to evaluate the derivative
     * @param dx spacing between two consecutive points
     * @param fct function to evaluate
     * @return the derivative of the function at x
     */
    template<class T>
    inline T derivative_3point_forward(T x, T dx, std::function<T(T)> &&fct) {
        return (-3 * fct(x) + 4 * fct(x + dx) - fct(x + 2 * dx)) / (2 * dx);
    }

    /**
     * @brief Compute the derivative of a function at x using a 3-point backward finite difference.
     *
     * This method computes the derivative using the points x, x - dx and x - 2*dx. This method
     * has a second-order accuracy.
     *
     * @param x point at which to evaluate the derivative
     * @param dx spacing between two consecutive points
     * @param fct function to evaluate
     * @return the derivative of the function at x
     */
    template<class T>
    inline T derivative_3point_backward(T x, T dx, std::function<T(T)> &&fct) {
        return (3 * fct(x) - 4 * fct(x - dx) + fct(x - 2 * dx)) / (2 * dx);
    }

    /**
     * @brief Compute the derivative of a function at x using a 5-point centered finite difference.
     *
     * This method computes the derivative using the points x, x + dx, x - dx, x + 2*dx and x -
     * 2*dx. This method has a fourth-order accuracy.
     *
     * @param x point at which to evaluate the derivative
     * @param dx spacing between two consecutive points
     * @param fct function to evaluate
     * @return the derivative of the function at x
     */
    template<class T>
    inline T derivative_5point_midpoint(T x, T dx, std::function<T(T)> &&fct) {
        return (-fct(x + 2 * dx) + 8 * fct(x + dx) - 8 * fct(x - dx) + fct(x - 2 * dx)) / (12 * dx);
    }

    /**
     * @brief Estimate the best step size for numerical differentiation of given order.
     *
     * @param order the order of the numerical differentiation
     * @return the estimated step size
     */
    template<class T>
    inline T estim_deriv_step(u32 order) {
        return sycl::powr(shambase::get_epsilon<T>(), 1.0 / (order + 1));
    }

} // namespace shammath
