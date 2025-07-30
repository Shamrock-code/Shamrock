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
 * @file CoordRange.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "intervals.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <limits>

namespace shammath {

    template<class T>
    struct CoordRange {

        using T_prop = shambase::VectorProperties<T>;

        T lower;
        T upper;

        inline CoordRange() = default;

        inline CoordRange(T lower, T upper) : lower(lower), upper(upper) {};

        inline CoordRange(std::tuple<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline CoordRange(std::pair<T, T> range)
            : lower(std::get<0>(range)), upper(std::get<1>(range)) {}

        inline T delt() const { return upper - lower; }

        inline CoordRange expand_all(typename T_prop::component_type value) {
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

        inline typename T_prop::component_type get_volume() {
            return sham::product_accumulate(upper - lower);
        }

        static CoordRange max_range();

        void check_throw_ranges(SourceLocation loc = SourceLocation{});

        inline CoordRange get_intersect(CoordRange other) const {
            return {sham::max(lower, other.lower), sham::min(upper, other.upper)};
        }

        inline CoordRange get_union(CoordRange other) const {
            return {sham::min(lower, other.lower), sham::max(upper, other.upper)};
        }

        inline bool contain_pos(T pos) { return is_in_half_open(pos, lower, upper); }

        inline bool is_not_empty() { return sham::vec_compare_geq(upper, lower); }

        inline CoordRange add_offset(T off) { return CoordRange{lower + off, upper + off}; }

        inline bool is_err_mode() {

            auto tmp = max_range();

            return sham::equals(tmp.lower, lower) && sham::equals(tmp.upper, upper);
        }
    };

    template<>
    inline CoordRange<f32_3> CoordRange<f32_3>::max_range() {

        CoordRange<f32_3> ret;

        ret.lower = {shambase::get_min<f32>(), shambase::get_min<f32>(), shambase::get_min<f32>()};

        ret.upper = {shambase::get_max<f32>(), shambase::get_max<f32>(), shambase::get_max<f32>()};

        return ret;
    }

    template<>
    inline CoordRange<f64_3> CoordRange<f64_3>::max_range() {

        CoordRange<f64_3> ret;

        ret.lower = {shambase::get_min<f64>(), shambase::get_min<f64>(), shambase::get_min<f64>()};

        ret.upper = {shambase::get_max<f64>(), shambase::get_max<f64>(), shambase::get_max<f64>()};

        return ret;
    }

    template<>
    inline CoordRange<u32_3> CoordRange<u32_3>::max_range() {

        CoordRange<u32_3> ret;

        ret.lower = {shambase::get_min<u32>(), shambase::get_min<u32>(), shambase::get_min<u32>()};

        ret.upper = {shambase::get_max<u32>(), shambase::get_max<u32>(), shambase::get_max<u32>()};

        return ret;
    }

    template<>
    inline CoordRange<u64_3> CoordRange<u64_3>::max_range() {

        CoordRange<u64_3> ret;

        ret.lower = {shambase::get_min<u64>(), shambase::get_min<u64>(), shambase::get_min<u64>()};

        ret.upper = {shambase::get_max<u64>(), shambase::get_max<u64>(), shambase::get_max<u64>()};

        return ret;
    }

    template<>
    inline CoordRange<i64_3> CoordRange<i64_3>::max_range() {

        CoordRange<i64_3> ret;

        ret.lower = {shambase::get_min<i64>(), shambase::get_min<i64>(), shambase::get_min<i64>()};

        ret.upper = {shambase::get_max<i64>(), shambase::get_max<i64>(), shambase::get_max<i64>()};

        return ret;
    }

} // namespace shammath
