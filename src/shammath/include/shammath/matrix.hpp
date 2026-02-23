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
 * @file matrix.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
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

        /// const overload
        inline constexpr auto get_mdspan() const {
            return std::mdspan<const T, std::extents<size_t, m, n>>(data.data());
        }

        /// Access the matrix entry at position (i, j)
        inline constexpr T &operator()(int i, int j) { return get_mdspan()(i, j); }

        /// const overload
        inline constexpr const T &operator()(int i, int j) const { return get_mdspan()(i, j); }

        /// Check if this matrix is equal to another one
        bool operator==(const mat<T, m, n> &other) const { return data == other.data; }

        inline mat &operator+=(const mat &other) {
#pragma unroll
            for (size_t i = 0; i < m * n; i++) {
                data[i] += other.data[i];
            }
            return *this;
        }

        /// check if this matrix is equal to another one at a given precison
        bool equal_at_precision(const mat<T, m, n> &other, const T precision) const {
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

template<class T, int m, int n>
struct sham::VectorProperties<shammath::mat<T, m, n>> {
    using component_type           = T;
    static constexpr u32 dimension = m * n;

    static constexpr bool is_float_based
        = std::is_same<T, f16>::value || std::is_same<T, f32>::value || std::is_same<T, f64>::value;
    static constexpr bool is_uint_based = std::is_same<T, u8>::value || std::is_same<T, u16>::value
                                          || std::is_same<T, u32>::value
                                          || std::is_same<T, u64>::value;
    static constexpr bool is_int_based = std::is_same<T, i8>::value || std::is_same<T, i16>::value
                                         || std::is_same<T, i32>::value
                                         || std::is_same<T, i64>::value;
    static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;

    static constexpr shammath::mat<T, m, n> get_min() {
        constexpr T min = shambase::get_min<T>();
        return {min};
    }
    static constexpr shammath::mat<T, m, n> get_max() {
        constexpr T max = shambase::get_max<T>();
        return {max};
    }
    static constexpr shammath::mat<T, m, n> get_zero() {
        constexpr T zero = 0;
        return {zero};
    }
};

using f32_3x3 = shammath::mat<f32, 3, 3>; ///< Alias for 3x3 float matrix
using f32_4x4 = shammath::mat<f32, 4, 4>; ///< Alias for 4x4 float matrix
using f64_3x3 = shammath::mat<f64, 3, 3>; ///< Alias for 3x3 double matrix
using f64_4x4 = shammath::mat<f64, 4, 4>; ///< Alias for 4x4 double matrix
