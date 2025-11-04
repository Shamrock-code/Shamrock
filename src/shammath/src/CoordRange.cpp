// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CoordRange.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shammath/CoordRange.hpp"

namespace shammath {

    template<class T>
    void throw_ill_formed(
        T lower, T upper, SourceLocation call, SourceLocation loc = SourceLocation{}) {
        throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
            "this range is ill formed normally upper > lower\n     lower = {}, upper = {}\n     "
            "call to check_throw = {}",
            lower,
            upper,
            call.format_multiline()));
    }

    template<>
    void CoordRange<f32_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<f64_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<u16_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<u32_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<u64_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<i32_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

    template<>
    void CoordRange<i64_3>::check_throw_ranges(SourceLocation loc) {
        if (lower.x() >= upper.x()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.y() >= upper.y()) {
            throw_ill_formed(lower, upper, loc);
        }
        if (lower.z() >= upper.z()) {
            throw_ill_formed(lower, upper, loc);
        }
    }

} // namespace shammath
