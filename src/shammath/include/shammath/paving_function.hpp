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
 * @file paving_function.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"

namespace shammath {

    template<typename Tvec, class paving_func>
    inline AABB<Tvec> f_aabb(
        const paving_func &paving, const AABB<Tvec> &aabb, int i, int j, int k) {

        static_assert(!paving_func::can_deform_aabb, "The paving function cannot deform the AABB");

        Tvec min = paving.f(aabb.lower, i, j, k);
        Tvec max = paving.f(aabb.upper, i, j, k);

        // if the min is greater than the max, swap them
        for (size_t d = 0; d < shambase::VectorProperties<Tvec>::dimension; ++d) {
            if (min[d] > max[d]) {
                std::swap(min[d], max[d]);
            }
        }

        return AABB<Tvec>{min, max};
    }

    template<typename Tvec, class paving_func>
    inline AABB<Tvec> f_aabb_inv(
        const paving_func &paving, const AABB<Tvec> &aabb, int i, int j, int k) {

        static_assert(!paving_func::can_deform_aabb, "The paving function cannot deform the AABB");

        Tvec min = paving.f_inv(aabb.lower, i, j, k);
        Tvec max = paving.f_inv(aabb.upper, i, j, k);

        // if the min is greater than the max, swap them
        for (size_t d = 0; d < shambase::VectorProperties<Tvec>::dimension; ++d) {
            if (min[d] > max[d]) {
                std::swap(min[d], max[d]);
            }
        }

        return AABB<Tvec>{min, max};
    }

    /**
     * @brief A structure for 3D paving functions with periodic boundary conditions.
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_periodic_3d {

        // only translation so f(AABB) is still an AABB
        inline static constexpr bool can_deform_aabb = false;

        Tvec box_size; ///< The size of the box in each dimension.

        /**
         * @brief Applies the paving function with periodic boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) const { return x + box_size * Tvec{i, j, k}; }

        /**
         * @brief Applies the inverse of the paving function.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) const { return x - box_size * Tvec{i, j, k}; }

        inline AABB<Tvec> f_aabb(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb(*this, aabb, i, j, k);
        }

        inline AABB<Tvec> f_aabb_inv(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb_inv(*this, aabb, i, j, k);
        }

        inline std::vector<std::array<int, 3>> get_paving_index_intersecting(
            AABB<Tvec> aabb) const {

            Tvec lower = aabb.lower / box_size;
            Tvec upper = aabb.upper / box_size;

            lower[0] = std::floor(lower[0]);
            lower[1] = std::floor(lower[1]);
            lower[2] = std::floor(lower[2]);

            i32_3 low_int = {
                static_cast<i32>(lower[0]), static_cast<i32>(lower[1]), static_cast<i32>(lower[2])};

            std::vector<std::array<int, 3>> indices;

            for (int i = low_int[0]; i <= upper[0]; ++i) {
                for (int j = low_int[1]; j <= upper[1]; ++j) {
                    for (int k = low_int[2]; k <= upper[2]; ++k) {
                        indices.push_back({i, j, k});
                    }
                }
            }

            return indices;
        }
    };

    /**
     * @brief A structure for 3D paving functions with general boundary conditions (periodic or
     * reflective per directions).
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_general_3d {

        // only translation so f(AABB) is still an AABB
        inline static constexpr bool can_deform_aabb = false;

        using Tscal = shambase::VecComponent<Tvec>;

        /**
         * @brief The size of the box in each dimension.
         */
        Tvec box_size;

        /**
         * @brief The center of the box in each dimension.
         */
        Tvec box_center;

        /**
         * @brief The boundary condition in each dimension.
         *
         * `true` means periodic boundary condition.
         * `false` means reflective boundary condition.
         */
        bool is_x_periodic;
        bool is_y_periodic;
        bool is_z_periodic;

        /**
         * @brief Applies the paving function with periodic or reflective boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) const {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off;
        }

        /**
         * @brief Applies the inverse of the paving function.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) const {
            Tvec tmp = x - box_size * Tvec{i, j, k};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }

        inline AABB<Tvec> f_aabb(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb(*this, aabb, i, j, k);
        }

        inline AABB<Tvec> f_aabb_inv(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb_inv(*this, aabb, i, j, k);
        }

        inline std::vector<std::array<int, 3>> get_paving_index_intersecting(
            AABB<Tvec> aabb) const {

            Tvec lower = aabb.lower / box_size;
            Tvec upper = aabb.upper / box_size;

            lower[0] = std::floor(lower[0]);
            lower[1] = std::floor(lower[1]);
            lower[2] = std::floor(lower[2]);

            i32_3 low_int = {
                static_cast<i32>(lower[0]), static_cast<i32>(lower[1]), static_cast<i32>(lower[2])};

            std::vector<std::array<int, 3>> indices;

            for (int i = low_int[0]; i <= upper[0]; ++i) {
                for (int j = low_int[1]; j <= upper[1]; ++j) {
                    for (int k = low_int[2]; k <= upper[2]; ++k) {
                        indices.push_back({i, j, k});
                    }
                }
            }

            return indices;
        }
    };

    /**
     * @brief A structure for 3D paving functions with shearing along the x-axis and general
     * boundary conditions.
     *
     * This structure supports both periodic and reflective boundary conditions in each dimension,
     * with additional shearing applied along the x-axis.
     *
     * @tparam Tvec A vector type.
     */
    template<typename Tvec>
    struct paving_function_general_3d_shear_x {

        // only translation so f(AABB) is still an AABB
        inline static constexpr bool can_deform_aabb = false;

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec box_size;   ///< The size of the box in each dimension.
        Tvec box_center; ///< The center of the box in each dimension.

        bool is_x_periodic; ///< Boundary condition for x dimension (true for periodic, false for
                            ///< reflective).
        bool is_y_periodic; ///< Boundary condition for y dimension (true for periodic, false for
                            ///< reflective).
        bool is_z_periodic; ///< Boundary condition for z dimension (true for periodic, false for
                            ///< reflective).

        Tscal shear_x; ///< Shearing factor applied along the x-axis.

        /**
         * @brief Applies the paving function with shearing and boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The transformed vector.
         */
        Tvec f(Tvec x, int i, int j, int k) const {
            Tvec off{
                (is_x_periodic) ? 0 : (x[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (x[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (x[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return x + box_size * Tvec{i, j, k} + off + shear_x * Tvec{j, 0, 0};
        }

        /**
         * @brief Applies the inverse of the paving function with shearing and boundary conditions.
         *
         * @param x The input vector.
         * @param i The periodic index along the x-axis.
         * @param j The periodic index along the y-axis.
         * @param k The periodic index along the z-axis.
         * @return The inverse transformed vector.
         */
        Tvec f_inv(Tvec x, int i, int j, int k) const {
            Tvec tmp = x - box_size * Tvec{i, j, k} - shear_x * Tvec{j, 0, 0};
            Tvec off{
                (is_x_periodic) ? 0 : (tmp[0] - box_center[0]) * (sham::m1pown<Tscal>(i) - 1),
                (is_y_periodic) ? 0 : (tmp[1] - box_center[1]) * (sham::m1pown<Tscal>(j) - 1),
                (is_z_periodic) ? 0 : (tmp[2] - box_center[2]) * (sham::m1pown<Tscal>(k) - 1)};
            return tmp + off;
        }

        inline AABB<Tvec> f_aabb(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb(*this, aabb, i, j, k);
        }

        inline AABB<Tvec> f_aabb_inv(const AABB<Tvec> &aabb, int i, int j, int k) const {
            return shammath::f_aabb_inv(*this, aabb, i, j, k);
        }

        inline std::vector<std::array<int, 3>> get_paving_index_intersecting(
            AABB<Tvec> aabb) const {

            Tvec lower = aabb.lower / box_size;
            Tvec upper = aabb.upper / box_size;

            lower[0] = std::floor(lower[0]);
            lower[1] = std::floor(lower[1]);
            lower[2] = std::floor(lower[2]);

            lower[0] -= 1;
            upper[0] += 1;

            i32_3 low_int = {
                static_cast<i32>(lower[0]), static_cast<i32>(lower[1]), static_cast<i32>(lower[2])};

            std::vector<std::array<int, 3>> indices;

            for (int i = low_int[0]; i <= upper[0]; ++i) {
                for (int j = low_int[1]; j <= upper[1]; ++j) {
                    for (int k = low_int[2]; k <= upper[2]; ++k) {

                        Tscal shear_x_floor = std::floor(j * shear_x / box_size[0]);
                        i32 s_x             = static_cast<i32>(shear_x_floor);

                        i32 i_s = i - s_x;

                        auto aabb_mapped = f_aabb(
                            AABB<Tvec>{box_center - box_size / 2, box_center + box_size / 2},
                            i_s,
                            j,
                            k);

                        shamcomm::logs::raw_ln("aabb: {}x{}", aabb.lower, aabb.upper);
                        shamcomm::logs::raw_ln(
                            "aabb_mapped: {}x{}", aabb_mapped.lower, aabb_mapped.upper);
                        shamcomm::logs::raw_ln(
                            "aabb_intersect: {}x{}",
                            aabb_mapped.get_intersect(aabb).lower,
                            aabb_mapped.get_intersect(aabb).upper);
                        shamcomm::logs::raw_ln(
                            "is_volume_not_null: {}",
                            aabb_mapped.get_intersect(aabb).is_volume_not_null());

                        if (aabb_mapped.get_intersect(aabb).is_volume_not_null()) {
                            indices.push_back({i_s, j, k});
                        }
                    }
                }
            }

            return indices;
        }
    };

} // namespace shammath
