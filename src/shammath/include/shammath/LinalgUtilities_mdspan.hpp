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

#include "shambackends/sycl.hpp"
#include <experimental/mdspan>
#include <array>

// the legendary trick to force a compilation error for missing ;
#define SHAM_ASSERT(x)                                                                             \
    do {                                                                                           \
    } while (false)

namespace shammath {

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_set_identity(const std::mdspan<T, Extents, Layout, Accessor> &input1) {

        SHAM_ASSERT(input1.extent(0) == input1.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                input1(i, j) = (i == j) ? 1 : 0;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_copy(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output) {

        SHAM_ASSERT(input.extent(0) == output.extent(0));
        SHAM_ASSERT(input.extent(1) == output.extent(1));

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                output(i, j) = input(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_add(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == output.extent(1));
        SHAM_ASSERT(input1.extent(0) == input2.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                output(i, j) = input1(i, j) + input2(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_sub(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == output.extent(1));
        SHAM_ASSERT(input1.extent(0) == input2.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                output(i, j) = input1(i, j) - input2(i, j);
            }
        }
    }

    template<
        class T,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_prod(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(0));
        SHAM_ASSERT(input2.extent(1) == output.extent(1));

        // output_ij = mat1_ik mat2_jk
        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input2.extent(1); j++) {
                T sum = 0;
                for (int k = 0; k < input1.extent(1); k++) {
                    sum += input1(i, k) * input2(k, j);
                }
                output(i, j) = sum;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void vec_copy(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output) {
        SHAM_ASSERT(input.extent(0) == output.extent(0));

        for (int i = 0; i < input.extent(0); i++) {
            output(i) = input(i);
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_set_nul(const std::mdspan<T, Extents, Layout, Accessor> &input) {
        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                input(i, j) = 0;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void vec_set_nul(const std::mdspan<T, Extents, Layout, Accessor> &input) {
        for (auto i = 0; i < input.extent(0); i++)
            input(i) = 0;
    }

    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_daxpy(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output,
        const U alpha,
        const U beta) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == output.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input1.extent(1); j++) {
                output(i, j) = alpha * input1(i, j) + beta * output(i, j);
            }
        }
    }

    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void vec_daxpy(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output,
        const U alpha,
        const U beta) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));

        for (int i = 0; i < input1.extent(0); i++) {
            output(i) = alpha * input1(i) + beta * output(i);
        }
    }

    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_gemm(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output,
        const U alpha,
        const U beta) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(0));
        SHAM_ASSERT(input2.extent(1) == output.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input2.extent(1); j++) {
                T sum = 0;
                for (int k = 0; k < input1.extent(1); k++) {
                    sum += input1(i, k) * input2(k, j);
                }
                output(i, j) = alpha * beta * sum;
            }
        }
    }

    template<class T, class U, class Extents, class Layout, class Accessor>
    inline void vec_scal(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output,
        const U scal) {
        SHAM_ASSERT(input.extent(0) == output.extent(0));

        for (int i = 0; i < input.extent(0); i++) {
            output(i) = scal * input(i);
        }
    }

    template<class T, class U, class Extents, class Layout, class Accessor>
    inline void mat_scal(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output,
        const U scal) {

        SHAM_ASSERT(input.extent(0) == output.extent(0));
        SHAM_ASSERT(input.extent(1) == output.extent(1));

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                output(i, j) = scal * input(i, j);
            }
        }
    }

    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_add_scal_id(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output,
        const U beta) {

        SHAM_ASSERT(input.extent(0) == output.extent(0));
        SHAM_ASSERT(input.extent(1) == output.extent(1));

        for (int i = 0; i < input.extent(0); i++) {
            output(i, i) = input(i, i) + beta;
        }
    }

    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Extents3,
        class Layout1,
        class Layout2,
        class Layout3,
        class Accessor1,
        class Accessor2,
        class Accessor3>
    inline void mat_gemv(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &M,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &x,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &y,
        const U alpha,
        const U beta) {

        SHAM_ASSERT(M.extent(1) == x.extent(0));
        SHAM_ASSERT(M.extent(0) == y.extent(0));

        for (int i = 0; i < M.extent(0); i++) {
            T sum = 0;
            for (int j = 0; j < M.extent(1); j++) {
                sum += M(i, j) * x(j);
            }
            y(i) = alpha * sum + beta * y(i);
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_L1_norm(const std::mdspan<T, Extents, Layout, Accessor> &M, T &res) {
        res = 0;
        for (auto i = 0; i < M.extent(0); i++) {
            T sum = 0;
            for (auto j = 0; j < M.extent(1); j++) {
                sum += abs(M(i, j));
            }
            res = sycl::max(res, sum);
        }
    }

    template<class T, int m, int n>
    class mat {
        public:
        std::array<T, m * n> data;
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, m, n>>(data.data());
        }

        inline constexpr T &operator()(int i, int j) { return get_mdspan()(i, j); }

        bool operator==(const mat<T, m, n> &other) { return data == other.data; }
    };

    template<class T, int n>
    class vect {
        public:
        std::array<T, n> data;
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, n>>(data.data());
        }

        inline constexpr T &operator()(int i, int j) { return get_mdspan()(i); }

        bool operator==(const vect<T, n> &other) { return data == other.data; }
    };

} // namespace shammath

/*
namespace shammath {

    template<class T, class Extents, class Layout, class Accessor>
    inline void copy_between_2d_mdspan(
        const std::mdspan<T, Extents, Layout, Accessor> &dest,
        const std::mdspan<T, Extents, Layout, Accessor> &src) {
        if (dest.extent(0) == src.extent(0) && dest.extent(1) == src.extent(1))
            for (auto i = 0; i < dest.extent(0); i++) {
                for (auto j = 0; j < dest.extent(1); j++)
                    dest(i, j) = src(i, j);
            }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void copy_between_1d_mdspan(
        const std::mdspan<T, Extents, Layout, Accessor> &dest,
        const std::mdspan<T, Extents, Layout, Accessor> &src) {
        if (dest.extent(0) == src.extent(0))
            for (auto i = 0; i < dest.extent(0); i++) {
                dest(i) = src(i);
            }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void set_nul_2d_mdspan(const std::mdspan<T, Extents, Layout, Accessor> &M) {
        for (auto i = 0; i < M.extent(0); i++) {
            for (auto j = 0; j < M.extent(1); j++) {
                M(i, j) = 0;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void set_nul_1d_mdspan(const std::mdspan<T, Extents, Layout, Accessor> &x) {
        for (auto i = 0; i < x.extent(0); i++)
            x(i) = 0;
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void set_nul_to_identity_2d_mdspan(const std::mdspan<T, Extents, Layout, Accessor> &M) {
        for (auto i = 0; i < M.extent(0); i++)
            M(i, i) = 1.0;
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mdspan_scalMat(const std::mdspan<T, Extents, Layout, Accessor> &M, const T scal) {
        for (auto i = 0; i < M.extent(0); i++) {
            for (auto j = 0; j < M.extent(1); j++)
                M(i, j) *= scal;
        }
    }

    template<class T, class Extents_2D, class Extents_1D, class Layout, class Accessor>
    inline void mdspan_MatVecMut(
        const std::mdspan<T, Extents_2D, Layout, Accessor> &M,
        const std::mdspan<T, Extents_1D, Layout, Accessor> &x,
        const std::mdspan<T, Extents_1D, Layout, Accessor> &res) {

        if (M.extent(1) == x.extent(0) && x.extent(0) == res.extent(0)) {
            for (auto i = 0; i < M.extent(0); i++) {
                T xi = 0;
                for (auto j = 0; j < M.extent(1); j++)
                    xi += M[i][j] * x[j];
                res[i] = xi;
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mdspan_MatMatAdd(
        const std::mdspan<T, Extents, Layout, Accessor> &M1,
        const std::mdspan<T, Extents, Layout, Accessor> &M2,
        const T alpha = 1.0 const T beta = 1.0) {
        if (M1.extent(0) == M2.extent(0) && M1.extent(1) == M2.extent(1)) {
            for (auto i = 0; i < M1.extent(0); i++) {
                for (auto j = 0; j < M1.extent(1); j++) {
                    M1(i, j) = alpha * M1(i, j) + beta * M2(i, j);
                }
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mdspan_VecVecAdd(
        const std::mdspan<T, Extents, Layout, Accessor> &x1,
        const std::mdspan<T, Extents, Layout, Accessor> &x2,
        const T alpha = 1.0,
        const T beta  = 1.0) {
        if (x1.extent(0) == x2.extent(0)) {
            for (auto i = 0; i < x1.extent(0); i++) {
                x1(i) = alpha * x1(i) + beta * x2(i);
            }
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void mdspan_MatMatMut(
        const std::mdspan<T, Extents, Layout, Accessor> &M1,
        const std::mdspan<T, Extents, Layout, Accessor> &M2,
        const std::mdspan<T, Extents, Layout, Accessor> &M,
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

    template<class T, class Extents, class Layout, class Accessor>
    inline void mdspan_L1_norm(const std::mdspan<T, Extents, Layout, Accessor> &M, T &res) {
        res = 0;
        for (auto i = 0; i < M.extent(0); i++) {
            T sum = 0;
            for (auto j = 0; j < M.extent(1); j++) {
                sum += sycl::abs(M(i, j));
            }
            res = sycl::max(res, sum);
        }
    }

    template<class T, class Extents, class Layout, class Accessor>
    inline void compute_add_id_scal(
        const std::mdspan<T, Extents, Layout, Accessor> &M,
        const T alpha = 1.,
        const T beta  = 1.0) {
        for (int i = 0; i < M.extent(0); i++) {
            M(i, i) = alpha * M(i, i) + beta;
        }
    }

} // namespace shammath

*/
