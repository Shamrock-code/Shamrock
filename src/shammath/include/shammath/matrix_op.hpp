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
 * @file matrix_op.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/math.hpp"
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
     * @brief Set the elements of a vector according to a user-provided function
     *
     * @param input The vector to update the elements of
     * @param func The function to use to update the elements of the matrix. The
     * function must take one argument being the index.
     *
     * @details The function `func` is called for each element of the vector, and
     * the value returned by the function is used to set the corresponding
     * element of the vector.
     */
    template<class T, class Extents, class Layout, class Accessor, class Func>
    inline void vec_set_vals(const std::mdspan<T, Extents, Layout, Accessor> &input, Func &&func) {

        shambase::check_functor_signature<T, int>(func);

        for (int i = 0; i < input.extent(0); i++) {
            input(i) = func(i);
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
    inline void mat_update_vals(
        const std::mdspan<T, Extents, Layout, Accessor> &input, Func &&func) {

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
    inline void mat_mul_scalar(
        const std::mdspan<T, Extents, Layout, Accessor> &input, const T &scalar) {
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
        class Ta,
        class Tb,
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
        const std::mdspan<Ta, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<Ta, Extents2, Layout2, Accessor2> &input2,
        const std::mdspan<Tb, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(0));
        SHAM_ASSERT(input2.extent(1) == output.extent(1));

        // output_ij = mat1_ik mat2_jk
        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input2.extent(1); j++) {
                Tb sum = 0;
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

    /**
     * @brief compute the L1 norm of a given matrix
     *
     * @param input The input matrix
     * @param res   The result of the L1 norm of input
     *
     * @details This function computes the compute the L1 norm of a given matrix and store
     * the result in res
     */
    template<class T, class U, class Extents, class Layout, class Accessor>
    inline void mat_L1_norm(const std::mdspan<T, Extents, Layout, Accessor> &input, U &res) {
        res = 0;
        for (auto i = 0; i < input.extent(0); i++) {
            T sum = 0;
            for (auto j = 0; j < input.extent(1); j++) {
                sum += abs(input(i, j));
            }
            res = sham::max(res, sum);
        }
    }

    /**
     * @brief Set the content of a matrix to zero
     *
     * @param input The matrix to set to the nul matrix
     */
    template<class T, class Extents, class Layout, class Accessor>
    inline void mat_set_nul(const std::mdspan<T, Extents, Layout, Accessor> &input) {
        mat_set_vals(input, [](auto i, auto j) -> T {
            return 0;
        });
    }

    /**
     * @brief Set the content of a vector to zero
     *
     * @param input the vector to set to zero
     */
    template<class T, class Extents, class Layout, class Accessor>
    inline void vec_set_nul(const std::mdspan<T, Extents, Layout, Accessor> &input) {
        for (auto i = 0; i < input.extent(0); i++) {
            input(i) = 0;
        }
    }

    /**
     * @brief Copy the content of one vector in another
     *
     * @param input the source vector
     * @param output the destination vector
     */
    template<class T, class Extents, class Layout, class Accessor>
    inline void vec_copy(
        const std::mdspan<T, Extents, Layout, Accessor> &input,
        const std::mdspan<T, Extents, Layout, Accessor> &output) {
        SHAM_ASSERT(input.extent(0) == output.extent(0));

        for (int i = 0; i < input.extent(0); i++) {
            output(i) = input(i);
        }
    }

    /**
     * @brief This function compute y = alpha*x + beta*y with x,y both vectors
     *
     * @param alpha a scalar ->
     * @param input a vector ->x
     * @param beta  a scalar
     * @param output a vector ->y
     */
    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void vec_axpy_beta(
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const U beta,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output) {

        SHAM_ASSERT(input.extent(0) == output.extent(0));

        for (int i = 0; i < input.extent(0); i++) {
            output(i) = alpha * input(i) + beta * output(i);
        }
    }

    /**
     * @brief This function compute y = alpha*x + y with x,y both vectors
     *
     * @param alpha a scalar ->
     * @param input a vector ->x
     * @param output a vector ->y
     */
    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void vec_axpy(
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output) {

        vec_axpy_beta(alpha, input, U{1}, output);
    }

    /**
     * @brief This function compute M = alpha*N + beta*M with M,N both matrices
     *
     * @param alpha a scalar
     * @param input a matrix -> N
     * @param beta  a scalar
     * @param output a matrix -> M
     */
    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_axpy_beta(
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const U beta,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output) {

        SHAM_ASSERT(input.extent(0) == output.extent(0));
        SHAM_ASSERT(input.extent(1) == output.extent(1));

        for (int i = 0; i < input.extent(0); i++) {
            for (int j = 0; j < input.extent(1); j++) {
                output(i, j) = alpha * input(i, j) + beta * output(i, j);
            }
        }
    }

    /**
     * @brief This function compute M = alpha*N + beta*M with M,N both matrices
     *
     * @param alpha a scalar
     * @param input a matrix -> N
     * @param output a matrix -> M
     */
    template<
        class T,
        class U,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_axpy(
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output) {

        mat_axpy_beta(alpha, input, U{1}, output);
    }

    /**
     * @brief This function compute C = alpha*A*B + beta*C with A,B,C are matrices
     *
     * @param alpha a scalar
     * @param input1 a matrix -> A
     * @param input2 a matrix -> B
     * @param beta  a scalar
     * @param output a matrix -> C
     */
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
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input1,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &input2,
        const U beta,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &output) {

        SHAM_ASSERT(input1.extent(0) == output.extent(0));
        SHAM_ASSERT(input1.extent(1) == input2.extent(0));
        SHAM_ASSERT(input2.extent(1) == output.extent(1));

        for (int i = 0; i < input1.extent(0); i++) {
            for (int j = 0; j < input2.extent(1); j++) {
                T sum = 0;
                for (int k = 0; k < input1.extent(1); k++) {
                    sum += input1(i, k) * input2(k, j);
                }
                output(i, j) = alpha * sum + beta * output(i, j);
            }
        }
    }

    /**
     * @brief This function compute addition of a matrix
     * with mutiple of identity matrix  A +=beta * I,
     *  where I is an Identity matrix. A is a matrix
     *
     * @param inout a matrix -> A
     * @param beta a scalar
     */

    template<class T, class U, class Extents1, class Layout1, class Accessor1>
    inline void mat_plus_equal_scalar_id(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &inout, const U beta) {
        for (int i = 0; i < inout.extent(0); i++) {
            inout(i, i) = inout(i, i) + beta;
        }
    }

    /**
     * @brief This function performs matrix-vector multiplication as y = a*Mx + b*y
     * @param alpha a scalar ->a
     * @param M a matrix
     * @param x vector
     * @param beta a scalar ->b
     * @param y a vector
     */
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
        const U alpha,
        const std::mdspan<T, Extents1, Layout1, Accessor1> &M,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &x,
        const U beta,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &y) {

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

    /**
     * @brief This function transposes a matrix
     * @param input matrix to transpose
     * @param output matrix to store the transposed matrix
     */
    template<
        class T,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void mat_transpose(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &input,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &output) {

        SHAM_ASSERT(input.extent(0) == output.extent(1));
        SHAM_ASSERT(input.extent(1) == output.extent(0));

        for (int i = 0; i < output.extent(0); i++) {
            for (int j = 0; j < output.extent(1); j++) {
                output(i, j) = input(j, i);
            }
        }
    }

    /**
     * @brief This function performs Cholesky decomposition. From a (real) symmetric,
     definite-positive square matrix $M$, return a lower triangular matrix $L$ such that
     \f[
        M = L L^T
     \f]
     * @param M a square symmetric, definite-positive matrix
     * @param L the output matrix to store the lower triangular matrix obtained by Cholesky
     decomposition
     */
    template<
        class T,
        class Extents1,
        class Extents2,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    inline void Cholesky_decomp(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &M,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &L) {

        SHAM_ASSERT(M.extent(1) == M.extent(0));
        SHAM_ASSERT(M.extent(0) == L.extent(0));
        SHAM_ASSERT(L.extent(1) == L.extent(0));

        for (int i = 0; i < M.extent(0); i++) {
            T sum_ik = 0.0;
            for (int k = 0; k < i; k++) {
                sum_ik += L(i, k) * L(i, k);
            }
            L(i, i) = std::sqrt(M(i, i) - sum_ik);
            for (int j = i + 1; j < M.extent(1); j++) {
                T sum_ikjk = 0.0;
                for (int k = 0; k < i; k++) {
                    sum_ikjk += L(i, k) * L(j, k);
                }
                L(j, i) = (M(i, j) - sum_ikjk) / L(i, i);
                L(i, j) = 0.0;
            }
        }
    }

    /**
* @brief This function solves a system of linear equations with Cholesky decomposition. The
system must have the form
\f[
    Mx = y
\f]
where $M$ is a (real) symmetric, definite-positive square matrix.
* @param M a square symmetric, definite-positive matrix
* @param y a vector, right hand side of the system
* @param x the ouput vector to store the solution of the system
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
    inline void Cholesky_solve(
        const std::mdspan<T, Extents1, Layout1, Accessor1> &M,
        const std::mdspan<T, Extents2, Layout2, Accessor2> &y,
        const std::mdspan<T, Extents3, Layout3, Accessor3> &x) {

        SHAM_ASSERT(M.extent(1) == M.extent(0));
        SHAM_ASSERT(M.extent(1) == x.extent(0));
        SHAM_ASSERT(M.extent(0) == y.extent(0));

        std::vector<T> a(M.extent(0));
        std::vector<T> L_storage(M.extent(0) * M.extent(1));

        std::mdspan<T, Extents1> L{L_storage.data(), M.extent(0), M.extent(1)};
        Cholesky_decomp(M, L);

        for (int i = 0; i < M.extent(0); i++) {
            T sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L(i, k) * a[k];
            }
            a[i] = (y(i) - sum) / L(i, i);
        }
        for (int i = M.extent(0) - 1; i >= 0; i--) {
            T sum = 0.0;
            for (int k = i + 1; k < M.extent(0); k++) {
                sum += L(k, i) * x(k);
            }
            x(i) = (a[i] - sum) / L(i, i);
        }
    }

} // namespace shammath
