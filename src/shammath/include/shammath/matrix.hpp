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
 * @file matrix.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/matrix_op.hpp"
#include <experimental/mdspan>
#include <array>

namespace shammath {

    /**
     * @brief Matrix class based on std::array storage and mdspan
     * @tparam T the type of the matrix entries
     * @tparam m the number of rows
     * @tparam n the number of columns
     */
    template<class T, int m, int n>
    class mat {
        public:
        /// The matrix data
        std::array<T, m * n> data;

        /// Get the matrix data as a mdspan
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, m, n>>(data.data());
        }

        /// Access the matrix entry at position (i, j)
        inline constexpr T &operator()(int i, int j) { return get_mdspan()(i, j); }

        /// Check if this matrix is equal to another one
        bool operator==(const mat<T, m, n> &other) { return data == other.data; }

        /// check if this matrix is equal to another one at a given precison
        bool equal_at_precision(const mat<T, m, n> &other, const T precision) {
            bool res = true;
            for (auto i = 0; i < m; i++) {
                for (auto j = 0; j < n; j++) {
                    if (sham::abs(data[i * n + j] - other.data[i * n + j]) >= precision) {
                        res = false;
                    }
                }
            }
            return res;
        }
    };

    /// Returns the identity matrix of size n
    template<class T, int n>
    inline constexpr mat<T, n, n> mat_identity() {
        mat<T, n, n> res{};
        mat_set_identity(res.get_mdspan());
        return res;
    }

    /**
     * @brief Vector class based on std::array storage and mdspan
     * @tparam T the type of the vector entries
     * @tparam n the number of entries
     */
    template<class T, int n>
    class vec {
        public:
        /// The vector data
        std::array<T, n> data;

        /// Get the vector data as a mdspan
        inline constexpr auto get_mdspan() {
            return std::mdspan<T, std::extents<size_t, n>>(data.data());
        }

        /// Get the vector data as a mdspan of a matrix with one column
        inline constexpr auto get_mdspan_mat_col() {
            return std::mdspan<T, std::extents<size_t, n, 1>>(data.data());
        }

        /// Get the vector data as a mdspan of a matrix with one row
        inline constexpr auto get_mdspan_mat_row() {
            return std::mdspan<T, std::extents<size_t, 1, n>>(data.data());
        }

        /// Access the vector entry at position i
        inline constexpr T &operator[](int i) { return get_mdspan()(i); }

        /// Check if this vector is equal to another one
        bool operator==(const vec<T, n> &other) { return data == other.data; }
    };

} // namespace shammath
