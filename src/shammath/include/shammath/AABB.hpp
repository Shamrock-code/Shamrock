// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

        using T_prop = shambase::VectorProperties<T>;
        using Tscal  = typename T_prop::component_type;

        T lower; ///< Lower bound of the AABB
        T upper; ///< Upper bound of the AABB

        inline AABB() = default;

        inline AABB(T lower, T upper) : lower(lower), upper(upper) {};

        inline AABB(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline AABB(std::pair<T, T> range) : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline Tscal get_volume() { return sham::product_accumulate(upper - lower); }

        inline T get_center() const noexcept { return (lower + upper) / 2; }

        inline T sum_bounds() const noexcept { return lower + upper; }

        inline AABB expand_all(typename T_prop::component_type value) {
            return AABB{lower - value, upper + value};
        }

        template<class Tb>
        inline AABB<Tb> convert() {
            using Tb_prop = shambase::VectorProperties<Tb>;
            static_assert(
                Tb_prop::dim == T_prop::dim, "you cannot change the dimension in convert");

            return {
                lower.template convert<Tb_prop::component_type>(),
                upper.template convert<Tb_prop::component_type>()};
        }

        inline AABB get_intersect(AABB other) const noexcept {
            return {sham::max(lower, other.lower), sham::min(upper, other.upper)};
        }

        [[nodiscard]] inline bool is_not_empty() const noexcept {
            return sham::vec_compare_geq(upper, lower);
        }

        [[nodiscard]] inline bool is_volume_not_null() const noexcept {
            return sham::vec_compare_g(upper, lower);
        }

        [[nodiscard]] inline bool is_surface() const noexcept {
            return sham::component_have_only_one_zero(delt()) && (is_not_empty());
        }

        [[nodiscard]] inline bool is_surface_or_volume() const noexcept {
            return sham::component_have_at_most_one_zero(delt()) && (is_not_empty());
        }

        [[nodiscard]] inline bool intersect_ray(Ray<T> ray) const noexcept {
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
    };

} // namespace shammath
