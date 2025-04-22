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
 * @file matrix_op.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/sycl.hpp"
#include <experimental/mdspan>
#include <array>

namespace shammath {

    /**
     * @brief Set the elements of a matrix according to a user-provided function
     *
     * @param input The matrix to set the elements of
     * @param func The function to use to set the elements of the matrix. The
     * function must take two arguments, the first being the row index and the
     * second being the column index. The function must return a value of type
     * `T`.
     *
     * @details The function `func` is called for each element of the matrix, and
     * the value returned by the function is used to set the corresponding
     * element of the matrix.
     */
    template<class T, class Extents, class Layout, class Accessor, class Func>
    inline void mat_set_vals(const std::mdspan<T, Extents, Layout, Accessor> &input, Func &&func) {

        shambase::check_functor_signature<T, int, int>(func);

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                input(i, j) = func(i, j);
            }
        }
    }

    /**
     * @brief Update the elements of a matrix according to a user-provided function
     *
     * @param input The matrix to update the elements of
     * @param func The function to use to update the elements of the matrix. The
     * function must take three arguments, the first being the value of the
     * element to update, the second being the row index and the third being the
     * column index.
     *
     * @details The function `func` is called for each element of the matrix, and
     * the value returned by the function is used to update the corresponding
     * element of the matrix.
     */
    template<class T, class Extents, class Layout, class Accessor, class Func>
    inline void
    mat_update_vals(const std::mdspan<T, Extents, Layout, Accessor> &input, Func &&func) {

        shambase::check_functor_signature<void, T &, int, int>(func);

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                func(input(i, j), i, j);
            }
        }
    }

    /**
     * @brief Set the content of a matrix to the identity matrix
     *
     * @param input1 The matrix to set to the identity matrix
     *
     * @note The matrix must be square.
     *
     * @details The identity matrix is a matrix with all elements on the main
     * diagonal (from the top-left to the bottom-right) set to 1, and all other
     * elements set to 0.
     */
    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_set_identity(const std::mdspan<T, Extents, Layout, Accessor> &input1) {

        SHAM_ASSERT(input1.extent(0) == input1.extent(1));

        mat_set_vals(input1, [](auto i, auto j) -> T {
            return (i == j) ? 1 : 0;
        });
    }

    /**
     * @brief Multiply a matrix by a scalar value
     *
     * @param input The matrix to multiply
     * @param scalar The scalar value to multiply by
     *
     * @details This function multiplies each element of the matrix by the scalar
     * value, and stores the result back in the matrix.
     */
    template<class T, class Extents, class Layout, class Accessor>
    inline void
    mat_mul_scalar(const std::mdspan<T, Extents, Layout, Accessor> &input, const T &scalar) {
        mat_update_vals(input, [&](T &v, auto i, auto j) {
            v *= scalar;
        });
    }

    /**
     * @brief Copy a matrix to another matrix
     *
     * @param input The matrix to copy
     * @param output The matrix to copy to
     *
     * @details This function copies each element of the input matrix to the
     * corresponding element of the output matrix.
     */
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

    /**
     * @brief Add two matrices element-wise.
     *
     * @param input1 The first input matrix.
     * @param input2 The second input matrix.
     * @param output The output matrix to store the result.
     *
     * @details This function performs element-wise addition of two matrices
     * and stores the result in the output matrix. The dimensions of both
     * input matrices and the output matrix must be the same.
     */
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
    inline void mat_plus(
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

    /**
     * @brief Add a matrix to another matrix element-wise and store the result in the first matrix.
     *
     * @param inout The matrix to be updated with the element-wise addition result.
     * @param matb The matrix to add to the first matrix.
     *
     * @details This function performs element-wise addition of the second matrix
     * to the first matrix, modifying the first matrix with the result. The matrices
     * must have the same dimensions.
     */
    template<
        class T,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_plus_equal(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &inout,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &matb) {

        SHAM_ASSERT(inout.extent(0) == inout.extent(0));
        SHAM_ASSERT(inout.extent(1) == inout.extent(1));

        for (int i = 0; i < inout.extent(0); i++) {
            for (int j = 0; j < inout.extent(1); j++) {
                inout(i, j) += matb(i, j);
            }
        }
    }

    /**
     * @brief Subtract two matrices element-wise.
     *
     * @param input1 The first input matrix.
     * @param input2 The second input matrix.
     * @param output The output matrix to store the result.
     *
     * @details This function performs element-wise subtraction of the second matrix
     * from the first matrix and stores the result in the output matrix. The dimensions
     * of both input matrices and the output matrix must be the same.
     */
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

    /**
     * @brief Subtract a matrix from another matrix element-wise and store the result in the first
     * matrix.
     *
     * @param inout The matrix to be updated with the element-wise subtraction result.
     * @param matb The matrix to subtract from the first matrix.
     *
     * @details This function performs element-wise subtraction of the second matrix
     * from the first matrix, modifying the first matrix with the result. The matrices
     * must have the same dimensions.
     */
    template<
        class T,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_sub_equal(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &inout,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &matb) {

        SHAM_ASSERT(inout.extent(0) == inout.extent(0));
        SHAM_ASSERT(inout.extent(1) == inout.extent(1));

        for (int i = 0; i < inout.extent(0); i++) {
            for (int j = 0; j < inout.extent(1); j++) {
                inout(i, j) -= matb(i, j);
            }
        }
    }

    /**
     * @brief Compute the product of two matrices.
     *
     * @param input1 The first input matrix.
     * @param input2 The second input matrix.
     * @param output The output matrix to store the result.
     *
     * @details This function computes the product of two matrices and stores
     * the result in the output matrix. The matrices must satisfy the
     * following conditions:
     *
     * - The number of rows of the first matrix must be equal to the number
     *   of rows of the output matrix.
     * - The number of columns of the first matrix must be equal to the
     *   number of rows of the second matrix.
     * - The number of columns of the second matrix must be equal to the
     *   number of columns of the output matrix.
     */
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

    /**
     * @brief Compute the inverse of a 3x3 matrix.
     *
     * @param input The input matrix to invert.
     * @param output The output matrix to store the result.
     *
     * @details This function computes the inverse of a 3x3 matrix and stores
     * the result in the output matrix.
     *
     * Note that this function assumes that the input matrix is invertible.
     * If the determinant of the matrix is zero, the function will produce an
     * invalid result.
     */
    template<class T, class SizeType, class Layout, class Accessor>
    inline void mat_inv_33(
        const std::mdspan<T, std::extents<SizeType, 3, 3>, Layout, Accessor> &input,
        const std::mdspan<T, std::extents<SizeType, 3, 3>, Layout, Accessor> &output) {

        T &a00 = input(0, 0);
        T &a10 = input(1, 0);
        T &a20 = input(2, 0);

        T &a01 = input(0, 1);
        T &a11 = input(1, 1);
        T &a21 = input(2, 1);

        T &a02 = input(0, 2);
        T &a12 = input(1, 2);
        T &a22 = input(2, 2);

        T det
            = (-a02 * a11 * a20 + a01 * a12 * a20 + a02 * a10 * a21 - a00 * a12 * a21
               - a01 * a10 * a22 + a00 * a11 * a22);

        output(0, 0) = (-a12 * a21 + a11 * a22) / det;
        output(1, 0) = (a12 * a20 - a10 * a22) / det;
        output(2, 0) = (-a11 * a20 + a10 * a21) / det;

        output(0, 1) = (a02 * a21 - a01 * a22) / det;
        output(1, 1) = (-a02 * a20 + a00 * a22) / det;
        output(2, 1) = (a01 * a20 - a00 * a21) / det;

        output(0, 2) = (-a02 * a11 + a01 * a12) / det;
        output(1, 2) = (a02 * a10 - a00 * a12) / det;
        output(2, 2) = (-a01 * a10 + a00 * a11) / det;
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

    template<class T, class Extents, class Layout, class Accessor>
    inline void vec_copy(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output) {
        SHAM_ASSERT(input.extent(0) == output.extent(0));

        for (int i = 0; i < input.extent(0); i++) {
            output(i) = input(i);
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

} // namespace shammath
