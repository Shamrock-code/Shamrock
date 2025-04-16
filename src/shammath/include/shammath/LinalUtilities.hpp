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
 * @file LinalUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include <cmath>

namespace shammath {

    const int Ndust = 2;
    using Array2D   = std::array<std::array<f64, Ndust + 1>, Ndust + 1>;
    using Array1D   = std::array<f64, Ndust + 1>;

    inline void copy_between_2d_array(Array2D &dest, Array2D &source, const size_t size) {
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                dest[i][j] = source[i][j];
    }

    inline void copy_between_1d_array(Array1D &dest, Array1D &source, const size_t size) {
        for (int i = 0; i < size; ++i)
            dest[i] = source[i];
    }

    inline void set_nul_1d_array(Array1D &x, const size_t size) {
        for (auto i = 0; i < size; i++)
            x[i] = 0;
    }

    inline void set_nul_2d_array(Array2D &M, const size_t size) {
        for (auto i = 0; i < size; i++)
            for (auto j = 0; j < size; j++)
                M[i][j] = 0;
    }

    inline void set_nul_to_identity_2d_array(Array2D &M, const size_t size) {
        for (auto i = 0; i < size; i++)
            M[i][i] = 1.0;
    }

    inline void compute_scalMat(Array2D &M, const f64 scal, const size_t size) {
        for (auto i = 0; i < size; i++) {
            for (auto j = 0; j < size; j++) {
                M[i][j] *= scal;
            }
        }
    }

    inline void compute_MatVecMut(const Array2D &M, Array1D &x, const size_t size) {
        Array1D tmp;
        copy_between_1d_array(tmp, x, size);
        for (auto i = 0; i < size; i++) {
            f64 xi = 0;
            for (auto j = 0; j < size; j++)
                xi += M[i][j] * tmp[j];
            x[i] = xi;
        }
    }

    inline void compute_MatMatAdd(
        Array2D &M1,
        const Array2D &M2,
        const size_t size,
        const f64 alpha = 1.0,
        const f64 beta  = 1.0) {
        for (auto i = 0; i < size; i++) {
            for (auto j = 0; j < size; j++) {
                f64 x    = alpha * M1[i][j] + beta * M2[i][j];
                M1[i][j] = x;
            }
        }
    }

    inline void compute_VecVecAdd(
        Array1D &x1,
        const Array1D &x2,
        const size_t size,
        const f64 alpha = 1.0,
        const f64 beta  = 1.0) {
        for (auto i = 0; i < size; i++) {
            x1[i] = alpha * x1[i] + beta * x2[i];
        }
    }

    inline void compute_MatMatMut(
        const Array2D &M1,
        Array2D &M2,
        const size_t size,
        const f64 alpha = 1.0,
        const f64 beta  = 1.0) {
        Array2D Res;
        copy_between_2d_array(Res, M2, size);
        for (auto i = 0; i < size; i++) {
            for (auto j = 0; j < size; j++) {
                f64 r = 0;
                for (auto k = 0; k < size; k++) {
                    r += M1[i][k] * Res[k][i];
                }

                M2[i][j] = alpha * beta * r;
            }
        }
    }

    inline void compute_L1_norm(const Array2D &M, const size_t size, f64 &res) {
        res = 0;
        for (int i = 0; i < size.i++) {
            f64 sum = 0;
            for (auto j = 0; j < size; j++) {
                sum += fabs(M[i][j]);
            }
            res = fmax(res, sum);
        }
    }

} // namespace shammath
