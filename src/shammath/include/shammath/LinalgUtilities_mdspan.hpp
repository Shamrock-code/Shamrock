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
 * @file LinalgUtilities_mdspan.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include "shammath/LinalUtilities.hpp"
#include <experimental/mdspan>

namespace shammath {
    namespace stdex = std::experimental;
    template<typename T>
    using mdspan2D
        = stdex::mdspan<T, stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent>>;

    template<typename T>
    using mdspan1D = stdex::mdspan<T, stdex::extents<size_t, stdex::dynamic_extent>>;

    template<typename T>
    inline void copy_between_2d_mdspan(mdspan2D<T> &dest, mdspan2D<T> &src) {
        if (dest.extent(0) == src.extent(0) && dest.extent(1) == src.extent(1))
            for (auto i = 0; i < dest.extent(0); i++) {
                for (auto j = 0; j < dest.extent(1); j++)
                    dest(i, j) = src(i, j);
            }
    }

    template<typename T>
    inline void copy_between_1d_mdspan(mdspan1D<T> &dest, mdspan1D<T> &src) {
        if (dest.extent(0) == src.extent(0))
            for (auto i = 0; i < dest.extent(0); i++) {
                dest(i) = src(i);
            }
    }

    template<typename T>
    inline void set_nul_2d_mdspan(mdspan2D<T> &M) {
        for (auto i = 0; i < M.extent(0); i++) {
            for (auto j = 0; j < M.extent(1); j++) {
                M(i, j) = 0;
            }
        }
    }

    template<typename T>
    inline void set_nul_1d_mdspan(mdspan1D<T> &x) {
        for (auto i = 0; i < x.extent(0); i++)
            x(i) = 0;
    }

    template<typename T>
    inline void set_nul_to_identity_2d_mdspan(mdspan2D<T> &M) {
        for (auto i = 0; i < M.extent(0); i++)
            M(i, i) = 1.0;
    }

    template<typename T>
    inline void mdspan_scalMat(mdspan2D<T> &M, const T scal) {
        for (auto i = 0; i < M.extent(0); i++) {
            for (auto j = 0; j < M.extent(1); j++)
                M(i, j) *= scal;
        }
    }

    template<typename T>
    inline void mdspan_MatVecMut(const mdspan2D<T> &M, const mdspan1D<T> &x, mdspan1D<T> &res) {
        copy_between_1d_mdspan(tmp, x);
        if (M.extent(1) == x.extent(0) && x.extent(0) == res.extent(0)) {
            for (auto i = 0; i < x.extent(0); i++) {
                f64 xi = 0;
                for (auto j = 0; j < size; j++)
                    xi += M[i][j] * x[j];
                res[i] = xi;
            }
        }
    }

    template<typename T>
    inline void mdspan_MatMatAdd(
        mdspan2D<T> &M1, const mdspan2D<T> &M2, const T alpha = 1.0 const T beta = 1.0) {
        if (M1.extent(0) == M2.extent(0) && M1.extent(1) == M2.extent(1)) {
            for (auto i = 0; i < M1.extent(0); i++) {
                for (auto j = 0; j < M1.extent(1); j++) {
                    M1(i, j) = alpha * M1(i, j) + beta * M2(i, j);
                }
            }
        }
    }

    template<typename T>
    inline void mdspan_VecVecAdd(
        mdspan1D<T> &x1, const mdspan1D<T> &x2, const T alpha = 1.0, const T beta = 1.0) {
        if (x1.extent(0) == x2.extent(0)) {
            for (auto i = 0; i < x1.extent(0); i++) {
                x1(i) = alpha * x1(i) + beta * x2(i);
            }
        }
    }

    template<typename T>
    inline void mdspan_MatMatMut(
        const mdspan2D<T> &M1,
        const mdspan2D<T> &M2,
        mdspan2D<T> &M,
        const T alpha = 1.0,
        const beta    = 1.0) {
        for (auto i = 0; i < M1.extent(0); i++) {
            for (auto j = 0; j < M2.extent(1); j++) {
                T r = 0;
                for (auto k = 0; k < M1.extent(1); k++) {
                    r += M1(i, k) * M2(k, j);
                }
                M(i, j) = alpha * beta * r;
            }
        }
    }

    template<typename T>
    inline void mdspan_L1_norm(const mdspan2D &M, T &res) {
        res = 0;
        for (auto i = 0; i < M.extent(0); i++) {
            T sum = 0;
            for (auto j = 0; j < M.extent(1); j++) {
                sum += sycl::abs(M(i, j));
            }
            res = sycl::max(res, sum);
        }
    }

    template<typename T>
    inline void compute_add_id_scal(mdspan2D<T> &M, const T alpha = 1., const T beta = 1.0) {
        for (int i = 0; i < M.extent(0); i++) {
            M(i, i) = alpha * M(i, i) + beta;
        }
    }

} // namespace shammath
