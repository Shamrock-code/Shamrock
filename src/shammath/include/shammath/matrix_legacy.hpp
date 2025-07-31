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
 * @file matrix_legacy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include <array>

namespace shammath {

    template<class T>
    inline auto
    compute_inv_33(std::array<sycl::vec<T, 3>, 3> mat) -> std::array<sycl::vec<T, 3>, 3> {

        using vec = sycl::vec<T, 3>;

        T a00 = mat[0].x();
        T a10 = mat[1].x();
        T a20 = mat[2].x();

        T a01 = mat[0].y();
        T a11 = mat[1].y();
        T a21 = mat[2].y();

        T a02 = mat[0].z();
        T a12 = mat[1].z();
        T a22 = mat[2].z();

        T det
            = (-a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
               - a01 * a10 * a22 + a00 * a11 * a22);

        return {
            (vec{-a12 * a21 + a11 * a22, a02 * a21 - a01 * a22, -a02 * a11 + a01 * a12} / det),
            (vec{a12 * a20 - a10 * a22, -a02 * a20 + a00 * a22, a02 * a10 - a00 * a12} / det),
            (vec{-a11 * a20 + a10 * a21, a01 * a20 - a00 * a21, -a01 * a10 + a00 * a11} / det)};
    }

    template<class T>
    inline auto
    mat_prod_33(std::array<sycl::vec<T, 3>, 3> mat_a, std::array<sycl::vec<T, 3>, 3> mat_b)
        -> std::array<sycl::vec<T, 3>, 3> {

        using vec = sycl::vec<T, 3>;

        T a00 = mat_a[0].x();
        T a10 = mat_a[1].x();
        T a20 = mat_a[2].x();

        T a01 = mat_a[0].y();
        T a11 = mat_a[1].y();
        T a21 = mat_a[2].y();

        T a02 = mat_a[0].z();
        T a12 = mat_a[1].z();
        T a22 = mat_a[2].z();

        T b00 = mat_b[0].x();
        T b10 = mat_b[1].x();
        T b20 = mat_b[2].x();

        T b01 = mat_b[0].y();
        T b11 = mat_b[1].y();
        T b21 = mat_b[2].y();

        T b02 = mat_b[0].z();
        T b12 = mat_b[1].z();
        T b22 = mat_b[2].z();

        return {
            vec{a00 * b00 + a01 * b10 + a02 * b20,
                a00 * b01 + a01 * b11 + a02 * b21,
                a00 * b02 + a01 * b12 + a02 * b22},
            vec{a10 * b00 + a11 * b10 + a12 * b20,
                a10 * b01 + a11 * b11 + a12 * b21,
                a10 * b02 + a11 * b12 + a12 * b22},
            vec{a20 * b00 + a21 * b10 + a22 * b20,
                a20 * b01 + a21 * b11 + a22 * b21,
                a20 * b02 + a21 * b12 + a22 * b22}};
    }
} // namespace shammath
