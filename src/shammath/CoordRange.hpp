// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once


/**
 * @file CoordRange.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/typeAliasVec.hpp"
#include "shambase/SourceLocation.hpp"
#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/vectors.hpp"
#include "intervals.hpp"

#include <limits>

namespace shammath {

    template<class T>
    struct CoordRange {


        using T_prop = shambase::VectorProperties<T>;

        T lower;
        T upper;

        inline CoordRange() = default;

        inline CoordRange(T lower, T upper) : lower(lower), upper(upper){};

        inline CoordRange(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline CoordRange(std::pair<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline CoordRange expand_all(typename T_prop::component_type value){
            return CoordRange{lower - value, upper + value};
        }

        inline void expand_center(T tol) {
            T center   = (lower + upper) / 2;
            T cur_delt = upper - lower;
            cur_delt /= 2;
            cur_delt *= tol;
            lower = center - cur_delt;
            upper = center + cur_delt;
        }

        inline typename T_prop::component_type get_volume(){
            return shambase::product_accumulate(upper - lower);
        }

        static CoordRange max_range();

        void check_throw_ranges(SourceLocation loc = SourceLocation{});

        inline CoordRange get_intersect(CoordRange other) const {
            return {
                shambase::sycl_utils::g_sycl_max(lower, other.lower),
                shambase::sycl_utils::g_sycl_min(upper, other.upper)
                };
        }

        inline CoordRange get_union(CoordRange other) const {
            return {
                shambase::sycl_utils::g_sycl_min(lower, other.lower),
                shambase::sycl_utils::g_sycl_max(upper, other.upper)
                };
        }

        inline bool contain_pos(T pos){
            return is_in_half_open(pos,lower,upper);
        }

        inline bool is_not_empty(){
            return shambase::vec_compare_geq(upper , lower);
        }

        inline CoordRange add_offset(T off){
            return CoordRange{lower + off, upper + off};
        }
        
        inline bool is_err_mode(){

            auto tmp = max_range();

            return shambase::vec_equals(tmp.lower , lower) && shambase::vec_equals(tmp.upper , upper);
        }
    };

    



    template<>
    inline CoordRange<f32_3> CoordRange<f32_3>::max_range() {

        CoordRange<f32_3> ret;

        ret.lower = {
            std::numeric_limits<f32>::min(),
            std::numeric_limits<f32>::min(),
            std::numeric_limits<f32>::min()};

        ret.upper = {
            std::numeric_limits<f32>::max(),
            std::numeric_limits<f32>::max(),
            std::numeric_limits<f32>::max()};

        return ret;
    }

    template<>
    inline CoordRange<f64_3> CoordRange<f64_3>::max_range() {

        CoordRange<f64_3> ret;

        ret.lower = {
            std::numeric_limits<f64>::min(),
            std::numeric_limits<f64>::min(),
            std::numeric_limits<f64>::min()};

        ret.upper = {
            std::numeric_limits<f64>::max(),
            std::numeric_limits<f64>::max(),
            std::numeric_limits<f64>::max()};

        return ret;
    }

    template<>
    inline CoordRange<u32_3> CoordRange<u32_3>::max_range() {

        CoordRange<u32_3> ret;

        ret.lower = {
            std::numeric_limits<u32>::min(),
            std::numeric_limits<u32>::min(),
            std::numeric_limits<u32>::min()};

        ret.upper = {
            std::numeric_limits<u32>::max(),
            std::numeric_limits<u32>::max(),
            std::numeric_limits<u32>::max()};

        return ret;
    }

    template<>
    inline CoordRange<u64_3> CoordRange<u64_3>::max_range() {

        CoordRange<u64_3> ret;

        ret.lower = {
            std::numeric_limits<u64>::min(),
            std::numeric_limits<u64>::min(),
            std::numeric_limits<u64>::min()};

        ret.upper = {
            std::numeric_limits<u64>::max(),
            std::numeric_limits<u64>::max(),
            std::numeric_limits<u64>::max()};

        return ret;
    }

    template<>
    inline CoordRange<i64_3> CoordRange<i64_3>::max_range() {

        CoordRange<i64_3> ret;

        ret.lower = {
            std::numeric_limits<i64>::min(),
            std::numeric_limits<i64>::min(),
            std::numeric_limits<i64>::min()};

        ret.upper = {
            std::numeric_limits<i64>::max(),
            std::numeric_limits<i64>::max(),
            std::numeric_limits<i64>::max()};

        return ret;
    }

} // namespace shammath