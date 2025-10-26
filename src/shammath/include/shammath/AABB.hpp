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
 * @file AABB.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/assert.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include <limits>

namespace shammath {

    template<class T>
    struct Ray {
        using T_prop = shambase::VectorProperties<T>;
        using Tscal  = typename T_prop::component_type;

        T origin;
        T direction;
        T inv_direction;

        inline Ray(T origin, T direction)
            : origin(origin), direction(direction), inv_direction(1 / direction) {

            Tscal f = sycl::length(direction);
            SHAM_ASSERT(f > 0);

            this->direction /= f;
            this->inv_direction *= f;
        }
    };

    /**
     * @brief Axis-Aligned bounding box
     *
     * This class describe a bounding box aligned on the axis.
     *
     * This class describe a domain of coordinates defined by a cartesian product of 1d ranges.
     * For exemple : [ax,bx] x [ay,by] x [az,bz]
     *
     * @tparam T Type of the coordinates
     */
    template<class T>
    struct AABB {

        using T_prop = shambase::VectorProperties<T>;   ///< Properties of the coordinates type
        using Tscal  = typename T_prop::component_type; ///< Scalar type of the coordinates

        T lower; ///< Lower bound of the AABB
        T upper; ///< Upper bound of the AABB

        inline AABB() = default;

        /**
         * @brief Construct an AABB from lower and upper bounds
         *
         * This constructor takes the lower and upper bounds of the AABB,
         * and constructs an AABB object from them.
         *
         * @param lower The lower bound of the AABB
         * @param upper The upper bound of the AABB
         */
        inline AABB(T lower, T upper)
            : lower(lower), upper(upper) {

              };

        /**
         * @brief Construct an AABB from a tuple of lower and upper bounds
         *
         * This constructor takes a tuple of lower and upper bounds, and constructs an AABB
         * object from them.
         *
         * @param range A tuple of lower and upper bounds
         */
        inline AABB(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        /**
         * @brief Construct an AABB from a pair of lower and upper bounds
         *
         * This constructor takes a pair of lower and upper bounds, and constructs an AABB
         * object from them.
         *
         * @param range A pair of lower and upper bounds
         */
        inline AABB(std::pair<T, T> range) : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        /**
         * @brief Returns the delta of the AABB
         *
         * This function returns the delta of the AABB, which is the difference between
         * the upper and lower bounds of the AABB.
         *
         * @return The delta of the AABB
         */
        inline T delt() const { return upper - lower; }

        /**
         * @brief Returns the volume of the AABB
         *
         * This function returns the volume of the AABB, which is the product of the differences
         * between the upper and lower bounds of the AABB for each axis.
         *
         * @return The volume of the AABB
         */
        inline Tscal get_volume() { return sham::product_accumulate(upper - lower); }

        /**
         * @brief Returns the center of the AABB
         *
         * This function returns the center of the AABB, which is the average of the lower and upper
         * bounds of the AABB.
         *
         * @return The center of the AABB
         */
        inline T get_center() const noexcept { return (lower + upper) / 2; }

        /**
         * @brief Returns the sum of the lower and upper bounds of the AABB
         *
         * This function returns the sum of the lower and upper bounds of the AABB.
         * This is useful for some algorithms, like the one used in the BVH tree.
         *
         * @return The sum of the lower and upper bounds of the AABB
         */
        inline T sum_bounds() const noexcept { return lower + upper; }

        /**
         * @brief Expand the AABB by a given value on all dimensions.
         *
         * This function expands the AABB by the given value on all dimensions.
         * The lower bound is shifted down by the given value, and the upper
         * bound is shifted up by the given value.
         *
         * @param value The value to expand the AABB by.
         *
         * @return A new AABB with the expanded bounds.
         */
        inline AABB expand_all(Tscal value) { return AABB{lower - value, upper + value}; }

        /**
         * @brief Converts the AABB to a different type.
         *
         * This function converts the current AABB to a new AABB with a different
         * coordinate type, specified by the template parameter Tb. The conversion
         * is performed for both the lower and upper bounds of the AABB.
         *
         * @tparam Tb The target coordinate type for conversion.
         *
         * @return A new AABB object with the converted coordinate type.
         *
         * @note The dimension of the AABB must remain the same during conversion.
         */
        template<class Tb>
        inline AABB<Tb> convert() {
            using Tb_prop = shambase::VectorProperties<Tb>;
            static_assert(
                Tb_prop::dim == T_prop::dim, "you cannot change the dimension in convert");

            return {
                lower.template convert<Tb_prop::component_type>(),
                upper.template convert<Tb_prop::component_type>()};
        }

        /**
         * @brief Compute the intersection of two AABB
         *
         * This function return a new AABB which is the intersection of the two AABB.
         * The intersection of two AABB is defined as the biggest AABB that is contained
         * in both AABB.
         *
         * @param other The other AABB
         *
         * @return The intersection of the two AABB
         */
        inline AABB get_intersect(AABB other) const noexcept {
            return {sham::max(lower, other.lower), sham::min(upper, other.upper)};
        }

        inline bool contains(AABB other) const noexcept {
            // return lower <= other.lower && upper >= other.upper;
            return sham::vec_compare_leq(lower, other.lower)
                   && sham::vec_compare_geq(upper, other.upper);
        }

        /**
         * @brief Checks if the AABB is non-empty.
         *
         * This function determines if the axis-aligned bounding box (AABB)
         * has a non-zero volume by comparing its upper and lower bounds.
         *
         * @return true if the AABB is non-empty, false otherwise.
         */
        [[nodiscard]] inline bool is_not_empty() const noexcept {
            return sham::vec_compare_geq(upper, lower);
        }

        /**
         * @brief Checks if the AABB has a non-zero volume.
         *
         * This function is more strict than is_not_empty() and checks if the AABB
         * has a non-zero volume by comparing its upper and lower bounds.
         *
         * @return true if the AABB has a non-zero volume, false otherwise.
         */
        [[nodiscard]] inline bool is_volume_not_null() const noexcept {
            return sham::vec_compare_g(upper, lower);
        }

        /**
         * @brief Checks if the AABB is a surface.
         *
         * This function determines if the axis-aligned bounding box (AABB)
         * is a surface. A surface is an AABB where only one of its dimensions
         * have a non-zero size.
         *
         * @return true if the AABB is a surface, false otherwise.
         */
        [[nodiscard]] inline bool is_surface() const noexcept {
            return sham::component_have_only_one_zero(delt()) && (is_not_empty());
        }

        /**
         * @brief Checks if the AABB is a surface or a volume.
         *
         * This function determines if the axis-aligned bounding box (AABB)
         * is a surface or a volume. A surface is an AABB where only one of its
         * dimensions have a non-zero size. A volume is an AABB where all its
         * dimensions have a non-zero size.
         *
         * @return true if the AABB is a surface or a volume, false otherwise.
         */
        [[nodiscard]] inline bool is_surface_or_volume() const noexcept {
            return sham::component_have_at_most_one_zero(delt()) && (is_not_empty());
        }

        /**
         * @brief Clamp a coordinate to the box
         *
         * This function clamp a coordinate to the box defined by the AABB.
         * It return the clamped value.
         *
         * @param[in] coord The coordinate to clamp
         * @return the clamped value
         */
        [[nodiscard]] inline T clamp_coord(T coord) const noexcept {
            return sycl::clamp(coord, lower, upper);
        }

        /**
         * @brief Check if the ray intersect the AABB
         *
         * This function perform a ray-AABB intersection test.
         * It return true if the ray intersect the AABB and false otherwise.
         *
         * @param[in] ray The ray to test
         * @return true if the ray intersect the AABB
         */
        [[nodiscard]] inline bool intersect_ray(Ray<T> ray) const noexcept;

        /// equal operator
        inline bool operator==(const AABB<T> &other) const noexcept {
            return sham::equals(lower, other.lower) && sham::equals(upper, other.upper);
        }

        /// not equal operator
        inline bool operator!=(const AABB<T> &other) const noexcept { return !(*this == other); }
    };

    template<class T>
    [[nodiscard]] inline bool AABB<T>::intersect_ray(Ray<T> ray) const noexcept {
        Tscal tmin = -shambase::get_infty<Tscal>(), tmax = shambase::get_infty<Tscal>();

        Tscal tx1 = (lower.x() - ray.origin.x()) * ray.inv_direction.x();
        Tscal tx2 = (upper.x() - ray.origin.x()) * ray.inv_direction.x();

        tmin = sycl::max(tmin, sycl::min(tx1, tx2));
        tmax = sycl::min(tmax, sycl::max(tx1, tx2));

        Tscal ty1 = (lower.y() - ray.origin.y()) * ray.inv_direction.y();
        Tscal ty2 = (upper.y() - ray.origin.y()) * ray.inv_direction.y();

        tmin = sycl::max(tmin, sycl::min(ty1, ty2));
        tmax = sycl::min(tmax, sycl::max(ty1, ty2));

        Tscal tz1 = (lower.z() - ray.origin.z()) * ray.inv_direction.z();
        Tscal tz2 = (upper.z() - ray.origin.z()) * ray.inv_direction.z();

        tmin = sycl::max(tmin, sycl::min(tz1, tz2));
        tmax = sycl::min(tmax, sycl::max(tz1, tz2));

        return tmax >= tmin;
    }

} // namespace shammath
