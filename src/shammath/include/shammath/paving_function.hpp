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

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shammath {

    template<typename Tvec>
    struct paving_function_periodic_3d {

        Tvec box_size;

        Tvec f(Tvec x, int i, int j, int k) { return x + box_size * Tvec{i, j, k}; }
        Tvec f_inv(Tvec x, int i, int j, int k) { return x - box_size * Tvec{i, j, k}; }
    };

    template<typename Tvec>
    struct paving_function_general_3d {

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec box_size;
        Tvec box_center;

        bool is_x_periodic;
        bool is_y_periodic;
        bool is_z_periodic;

        Tvec f(Tvec x, int i, int j, int k) {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off;
        }

        Tvec f_inv(Tvec x, int i, int j, int k) {
            Tvec tmp = x - box_size * Tvec{i, j, k};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }
    };

    template<typename Tvec>
    struct paving_function_general_3d_shear_x {

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec box_size;
        Tvec box_center;

        bool is_x_periodic;
        bool is_y_periodic;
        bool is_z_periodic;

        Tscal shear_x;

        Tvec f(Tvec x, int i, int j, int k) {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off + shear_x * Tvec{j, 0, 0};
        }

        Tvec f_inv(Tvec x, int i, int j, int k) {
            Tvec tmp = x - box_size * Tvec{i, j, k} - shear_x * Tvec{j, 0, 0};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }
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
