// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CoordRangeTransform.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/CoordRangeTransform.hpp"
#include <stdexcept>

namespace shammath {

    void check_divisor_throw(u64_3 val, u64_3 divisor) {
        bool cmp_x = val.x() % divisor.x() == 0;
        bool cmp_y = val.y() % divisor.y() == 0;
        bool cmp_z = val.z() % divisor.z() == 0;

        if (!cmp_x) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the divisor does not divide the value on component x\n"
                "  val     = ("
                + std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", "
                + std::to_string(val.z()) + ")\n" + "  divisor = (" + std::to_string(divisor.x())
                + ", " + std::to_string(divisor.y()) + ", " + std::to_string(divisor.z()) + ")\n");
        }

        if (!cmp_y) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the divisor does not divide the value on component y\n"
                "  val     = ("
                + std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", "
                + std::to_string(val.z()) + ")\n" + "  divisor = (" + std::to_string(divisor.x())
                + ", " + std::to_string(divisor.y()) + ", " + std::to_string(divisor.z()) + ")\n");
        }

        if (!cmp_z) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the divisor does not divide the value on component z\n"
                "  val     = ("
                + std::to_string(val.x()) + ", " + std::to_string(val.y()) + ", "
                + std::to_string(val.z()) + ")\n" + "  divisor = (" + std::to_string(divisor.x())
                + ", " + std::to_string(divisor.y()) + ", " + std::to_string(divisor.z()) + ")\n");
        }
    }

    template<class Ta, class Tb>
    void check_divisor_throw(Ta val, Tb divisor) {
        check_divisor_throw(val.template convert<u64>(), divisor.template convert<u64>());
    }

    template<class Tsource, class Tdest>
    void CoordRangeTransform<Tsource, Tdest>::print_transform() const {
        if (mode == multiply) {
            shamlog_debug_ln(
                "CoordRangeTransform", "multiply:", fact, source_coord_min, dest_coord_min);
        } else {
            shamlog_debug_ln(
                "CoordRangeTransform", "divide  :", fact, source_coord_min, dest_coord_min);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // constructor implementation
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<>
    CoordRangeTransform<u64_3, f32_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<f32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u64_3 source_delt = source_range.delt();
        f32_3 dest_delt   = dest_range.delt();

        f32_3 source_sz_conv = source_delt.convert<f32>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u64_3, f64_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<f64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u64_3 source_delt = source_range.delt();
        f64_3 dest_delt   = dest_range.delt();

        f64_3 source_sz_conv = source_delt.convert<f64>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u32_3, f32_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<f32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u32_3 source_delt = source_range.delt();
        f32_3 dest_delt   = dest_range.delt();

        f32_3 source_sz_conv = source_delt.convert<f32>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u32_3, f64_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<f64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u32_3 source_delt = source_range.delt();
        f64_3 dest_delt   = dest_range.delt();

        f64_3 source_sz_conv = source_delt.convert<f64>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u16_3, f32_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<f32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u16_3 source_delt = source_range.delt();
        f32_3 dest_delt   = dest_range.delt();

        f32_3 source_sz_conv = source_delt.convert<f32>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u16_3, f64_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<f64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        mode = multiply;

        u16_3 source_delt = source_range.delt();
        f64_3 dest_delt   = dest_range.delt();

        f64_3 source_sz_conv = source_delt.convert<f64>();

        fact = dest_delt / source_sz_conv;
    }

    template<>
    CoordRangeTransform<u64_3, u64_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<u64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u64_3 source_delt = source_range.delt();
        u64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = dest_delt / source_delt;
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = source_delt / dest_delt;
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<i64_3, u64_3>::CoordRangeTransform(
        CoordRange<i64_3> source_range, CoordRange<u64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u64_3 source_delt = source_range.delt().convert<u64>();
        u64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = dest_delt / source_delt;
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = source_delt / dest_delt;
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u64_3, i64_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<i64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        i64_3 source_delt = source_range.delt().convert<i64>();
        i64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = dest_delt / source_delt;
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = source_delt / dest_delt;
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u64_3, u32_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<u32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u64_3 source_delt = source_range.delt();
        u32_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt.convert<u64>() / source_delt).convert<u32>();
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt / dest_delt.convert<u64>()).convert<u32>();
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u64_3, i32_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<i32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u64_3 source_delt = source_range.delt();
        i32_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt.convert<u64>() / source_delt).convert<i32>();
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt / dest_delt.convert<u64>()).convert<i32>();
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u32_3, u16_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<u16_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u32_3 source_delt = source_range.delt();
        u16_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt.convert<u32>() / source_delt).convert<u16>();
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt / dest_delt.convert<u32>()).convert<u16>();
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u32_3, u64_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<u64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u32_3 source_delt = source_range.delt();
        u64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<u64>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<u64>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u32_3, i64_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<i64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u32_3 source_delt = source_range.delt();
        i64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<i64>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<i64>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u32_3, i32_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<i32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u32_3 source_delt = source_range.delt();
        i32_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<i32>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<i32>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u16_3, u64_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<u64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u16_3 source_delt = source_range.delt();
        u64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<u64>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<u64>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u16_3, u32_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<u32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u16_3 source_delt = source_range.delt();
        u32_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<u32>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<u32>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u16_3, i32_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<i32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u16_3 source_delt = source_range.delt();
        i32_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<i32>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<i32>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u16_3, i64_3>::CoordRangeTransform(
        CoordRange<u16_3> source_range, CoordRange<i64_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        source_range.check_throw_ranges();
        dest_range.check_throw_ranges();

        u16_3 source_delt = source_range.delt();
        i64_3 dest_delt   = dest_range.delt();

        bool cmp_x = dest_delt.x() >= source_delt.x();
        bool cmp_y = dest_delt.y() >= source_delt.y();
        bool cmp_z = dest_delt.z() >= source_delt.z();

        if ((cmp_x == cmp_y) && (cmp_z == cmp_y)) {

            bool obj_greater_than_patch = cmp_x;

            if (obj_greater_than_patch) {
                check_divisor_throw(dest_delt, source_delt);
                mode = multiply;
                fact = (dest_delt / source_delt.convert<i64>());
            } else {
                check_divisor_throw(source_delt, dest_delt);
                mode = divide;
                fact = (source_delt.convert<i64>() / dest_delt);
            }

        } else {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the range comparaison are not the same");
        }
    }

    template<>
    CoordRangeTransform<u64_3, u16_3>::CoordRangeTransform(
        CoordRange<u64_3> source_range, CoordRange<u16_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "this coordinate conversion mode is not implemented");
    }

    template<>
    CoordRangeTransform<u32_3, u32_3>::CoordRangeTransform(
        CoordRange<u32_3> source_range, CoordRange<u32_3> dest_range)
        : source_coord_min(source_range.lower), dest_coord_min(dest_range.lower) {

        throw shambase::make_except_with_loc<std::invalid_argument>(
            "this coordinate conversion mode is not implemented");
    }

    template class CoordRangeTransform<u64_3, f32_3>;
    template class CoordRangeTransform<u64_3, f64_3>;
    template class CoordRangeTransform<u64_3, u64_3>;
    template class CoordRangeTransform<u64_3, u32_3>;
    template class CoordRangeTransform<u64_3, i64_3>;
    template class CoordRangeTransform<u64_3, i32_3>;
    template class CoordRangeTransform<u64_3, u16_3>;

    template class CoordRangeTransform<u32_3, f32_3>;
    template class CoordRangeTransform<u32_3, f64_3>;
    template class CoordRangeTransform<u32_3, u64_3>;
    template class CoordRangeTransform<u32_3, u32_3>;
    template class CoordRangeTransform<u32_3, i64_3>;
    template class CoordRangeTransform<u32_3, i32_3>;
    template class CoordRangeTransform<u32_3, u16_3>;

    template class CoordRangeTransform<u16_3, f32_3>;
    template class CoordRangeTransform<u16_3, f64_3>;
    template class CoordRangeTransform<u16_3, u64_3>;
    template class CoordRangeTransform<u16_3, u32_3>;
    template class CoordRangeTransform<u16_3, i64_3>;
    template class CoordRangeTransform<u16_3, i32_3>;
    template class CoordRangeTransform<u16_3, u16_3>;

} // namespace shammath
