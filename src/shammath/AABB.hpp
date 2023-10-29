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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/SourceLocation.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"

#include <limits>

namespace shammath {

    /**
     * @brief Axis-Aligned bounding box
     * This class describe a bounding box aligned on the axis
     * This class describe a domain of coordinates defined by a cartesian product of 1d ranges
     * For exemple : [ax,bx] x [ay,by] x [az,bz]
     * 
     * @tparam T 
     */
    template<class T>
    struct AABB {

        using T_prop = shambase::VectorProperties<T>;

        T lower;
        T upper;

        inline AABB() = default;

        inline AABB(T lower, T upper) : lower(lower), upper(upper){};

        inline AABB(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline AABB(std::pair<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline typename T_prop::component_type get_volume(){
            return shambase::product_accumulate(upper - lower);
        }

        inline AABB get_intersect(AABB other) const {
            return {
                shambase::sycl_utils::g_sycl_max(lower, other.lower),
                shambase::sycl_utils::g_sycl_min(upper, other.upper)
                };
        }

        inline bool is_not_empty(){
            return shambase::vec_compare_geq(upper , lower);
        }

        inline bool is_surface(){
            return shambase::component_have_only_one_zero(delt()) && (is_not_empty());
        }

        inline bool is_surface_or_volume(){
            return shambase::component_have_at_most_one_zero(delt()) && (is_not_empty());
        }
    };

}