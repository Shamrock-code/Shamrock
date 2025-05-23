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
 * @file paving_function.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

namespace shammath {

    template<typename T>
    struct paving_function_periodic_3d {

        T box_size;

        T f(T x, int i, int j, int k) { return x + box_size * T{i, j, k}; }
        T f_inv(T x, int i, int j, int k) { return x - box_size * T{i, j, k}; }
    };

    template<typename T>
    struct paving_function_general_3d {

        T box_size;

        bool is_x_periodic;
        bool is_y_periodic;
        bool is_z_periodic;

        T f(T x, int i, int j, int k) { return x + box_size * T{i, j, k}; }
        T f_inv(T x, int i, int j, int k) { return x - box_size * T{i, j, k}; }
    };

    /*
    def f_pavving_reflection(x, y, i, j):
        center_x = box_size_x * 0.5
        center_y = box_size_y * 0.5

        res_x = x + i * box_size_x + shear_x*j
        res_y = y + j * box_size_y

        if not is_x_periodic:
            res_x += (x - center_x) * ( (-1)**i - 1)

        if not is_y_periodic:
            res_y += (y - center_y) * ( (-1)**j - 1)

        return (
            res_x,
            res_y
        )


    def f_pavving_reflection_inv(x, y, i, j):
        center_x = box_size_x * 0.5
        center_y = box_size_y * 0.5

        res_x = x - i * box_size_x - shear_x*j
        res_y = y - j * box_size_y

        if not is_x_periodic:
            res_x += (res_x - center_x) * ( (-1)**i - 1)

        if not is_y_periodic:
            res_y += (res_y - center_y) * ( (-1)**j - 1)

        return (
            res_x,
            res_y
        )
    */

} // namespace shammath
