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
 * @file format.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <source_location>
#include <string_view>

namespace shambase {

    /**
     * @brief Formatter alias for `fmt::formatter`
     *
     * This alias is used to prevent explicit use of the `fmt` library in the
     * codebase. This way, we can change the formatting library without having
     * to modify all the code that uses it.
     *
     * @tparam T Type to format
     */
    template<class T>
    using formatter = fmt::formatter<T>;

    /// format error callback
    fmt::format_error make_format_exception(
        std::string_view function_call,
        std::string_view what,
        const std::string &fmt_string,
        std::source_location loc = std::source_location::current());

    using format_except_builder_t = fmt::format_error (*)(
        std::string_view, std::string_view, const std::string &, std::source_location);

    void set_format_exception_builder(format_except_builder_t callback);

    inline __attribute__((always_inline)) auto vformat(std::string_view fmt, fmt::format_args args)
        -> std::string {
        try {
            return fmt::vformat(fmt, args);
        } catch (const std::exception &e) {
            throw make_format_exception("vformat", e.what(), std::string(fmt));
        }
    }

    inline __attribute__((always_inline)) auto vformat(fmt::string_view fmt, fmt::format_args args)
        -> std::string {
        try {
            return fmt::vformat(fmt, args);
        } catch (const std::exception &e) {
            throw make_format_exception("vformat", e.what(), fmt::to_string(fmt));
        }
    }

    /**
     * @brief format a string using fmtlib style
     * Cheat sheet : https://hackingcpp.com/cpp/libs/fmt.html
     *
     * @tparam T
     * @param fmt the format string
     * @param args the arguments to format against
     * @return std::string the formatted string
     */
    template<typename... T>
    inline __attribute__((always_inline)) auto format(fmt::format_string<T...> fmt, T &&...args)
        -> std::string {
        return shambase::vformat(fmt, fmt::make_format_args(args...));
    }

    /**
     * @brief format a string using C printf style
     * https://cplusplus.com/reference/cstdio/printf/
     *
     * @tparam T
     * @param fmt the format string
     * @param args the arguments to format against
     * @return std::string the formatted string
     */
    template<typename... T>
    inline std::string format_printf(std::string_view format, const T &...args) {
        try {
            return fmt::sprintf(format, args...);
        } catch (const std::exception &e) {
            throw make_format_exception("printf", e.what(), std::string(format));
        }
    }
} // namespace shambase
