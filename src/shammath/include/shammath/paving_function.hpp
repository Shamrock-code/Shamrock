// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file paving_function.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shammath {

    /**
     * @brief A structure for 3D paving functions with periodic boundary conditions.
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_periodic_3d {

        Tvec box_size; ///< The size of the box in each dimension.

        /**
         * @brief Applies the paving function with periodic boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) { return x + box_size * Tvec{i, j, k}; }

        /**
         * @brief Applies the inverse of the paving function.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) { return x - box_size * Tvec{i, j, k}; }
    };

    /**
     * @brief A structure for 3D paving functions with general boundary conditions (periodic or
     * reflective per directions).
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_general_3d {

        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @brief The size of the box in each dimension.
         */
        Tvec box_size;

        /**
         * @brief The center of the box in each dimension.
         */
        Tvec box_center;

        /**
         * @brief The boundary condition in each dimension.
         *
         * `true` means periodic boundary condition.
         * `false` means reflective boundary condition.
         */
        bool is_x_periodic;
        bool is_y_periodic;
        bool is_z_periodic;

        /**
         * @brief Applies the paving function with periodic or reflective boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off;
        }

        /**
         * @brief Applies the inverse of the paving function.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) {
            Tvec tmp = x - box_size * Tvec{i, j, k};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }
    };

    /**
     * @brief A structure for 3D paving functions with shearing along the x-axis and general
     * boundary conditions.
     *
     * This structure supports both periodic and reflective boundary conditions in each dimension,
     * with additional shearing applied along the x-axis.
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_general_3d_shear_x {

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec box_size;   ///< The size of the box in each dimension.
        Tvec box_center; ///< The center of the box in each dimension.

        bool is_x_periodic; ///< Boundary condition for x dimension (true for periodic, false for
                            ///< reflective).
        bool is_y_periodic; ///< Boundary condition for y dimension (true for periodic, false for
                            ///< reflective).
        bool is_z_periodic; ///< Boundary condition for z dimension (true for periodic, false for
                            ///< reflective).

        Tscal shear_x; ///< Shearing factor applied along the x-axis.

        /**
         * @brief Applies the paving function with shearing and boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off + shear_x * Tvec{j, 0, 0};
        }

        /**
         * @brief Applies the inverse of the paving function with shearing and boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) {
            Tvec tmp = x - box_size * Tvec{i, j, k} - shear_x * Tvec{j, 0, 0};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }
    };

} // namespace shammath
