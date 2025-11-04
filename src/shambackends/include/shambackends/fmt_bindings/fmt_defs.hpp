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
 * @file fmt_defs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Define the fmt formatters for sycl::vec
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

#ifndef DOXYGEN
template<class T>
struct fmt::formatter<sycl::vec<T, 2>> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(sycl::vec<T, 2> c, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({},{})", c.x(), c.y());
    }
};

template<class T>
struct fmt::formatter<sycl::vec<T, 3>> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(sycl::vec<T, 3> c, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({},{},{})", c.x(), c.y(), c.z());
    }
};

template<class T>
struct fmt::formatter<sycl::vec<T, 4>> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(sycl::vec<T, 4> c, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({},{},{},{})", c.x(), c.y(), c.z(), c.w());
    }
};

template<class T>
struct fmt::formatter<sycl::vec<T, 8>> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(sycl::vec<T, 8> c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(),
            "({},{},{},{},{},{},{},{})",
            c.s0(),
            c.s1(),
            c.s2(),
            c.s3(),
            c.s4(),
            c.s5(),
            c.s6(),
            c.s7());
    }
};

template<class T>
struct fmt::formatter<sycl::vec<T, 16>> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(sycl::vec<T, 16> c, FormatContext &ctx) const {
        return fmt::format_to(
            ctx.out(),
            "(({},{},{},{}),({},{},{},{}),({},{},{},{}),({},{},{},{}))",
            c.s0(),
            c.s1(),
            c.s2(),
            c.s3(),
            c.s4(),
            c.s5(),
            c.s6(),
            c.s7(),
            c.s8(),
            c.s9(),
            c.sA(),
            c.sB(),
            c.sC(),
            c.sD(),
            c.sE(),
            c.sF());
    }
};

#endif
