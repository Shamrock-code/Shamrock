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
 * @brief Core formatting functions: `format`, `vformat`, and `format_printf`
 *
 * This is the primary entry point for the shamformat library. It re-exports
 * aliases and exception declarations from the other headers and provides the
 * public formatting API wrapped with exception handling.
 */

#include "sham/format/aliases.hpp" // IWYU pragma: export
#include "sham/format/format_exception.hpp"
#include <fmt/printf.h>

namespace sham {

    /**
     * @brief Low-level formatted string output given pre-packaged arguments.
     *
     * Wraps `fmt::vformat` with exception handling that delegates to
     * `make_format_exception` on errors.
     *
     * @param fmt the format string
     * @param args pre-packaged format arguments
     * @return std::string the formatted output
     * @throws sham::format_error on formatting errors
     */
    inline __attribute__((always_inline)) auto vformat(std::string_view fmt, fmt::format_args args)
        -> std::string {
        try {
            return fmt::vformat(fmt, args);
        } catch (const std::exception &e) {
            throw make_format_exception("vformat", e.what(), std::string(fmt));
        }
    }

    /**
     * @brief Overload accepting a `fmt::string_view` format string.
     *
     * Wraps `fmt::vformat` with exception handling that delegates to
     * `make_format_exception` on errors.
     *
     * @param fmt the format string
     * @param args pre-packaged format arguments
     * @return std::string the formatted output
     * @throws sham::format_error on formatting errors
     */
    inline __attribute__((always_inline)) auto vformat(fmt::string_view fmt, fmt::format_args args)
        -> std::string {
        try {
            return fmt::vformat(fmt, args);
        } catch (const std::exception &e) {
            throw make_format_exception("vformat", e.what(), fmt::to_string(fmt));
        }
    }

    /**
     * @brief Format a string using fmt/lib-style format specifiers.
     *
     * Wraps `fmt::format` with exception handling that delegates to
     * `make_format_exception` on errors. This is the primary formatting
     * API; see https://fmt.dev/latest/syntax.html for format string syntax.
     *
     * @tparam T argument types accepted by the format string
     * @param fmt the format string (must be compile-time constant)
     * @param args the arguments to insert into the format string
     * @return std::string the formatted output
     * @throws sham::format_error on formatting errors
     */
    template<typename... T>
    inline __attribute__((always_inline)) auto format(fmt::format_string<T...> fmt, T &&...args)
        -> std::string {
        return sham::vformat(fmt, fmt::make_format_args(args...));
    }

    /**
     * @brief Format a string using C `printf`-style format specifiers.
     *
     * Wraps `fmt::sprintf` with exception handling that delegates to
     * `make_format_exception` on errors. See
     * https://fmt.dev/latest/syntax.html for supported specifiers.
     *
     * @tparam T argument types accepted by the format string
     * @param format the printf-style format string
     * @param args the arguments to insert into the format string
     * @return std::string the formatted output
     * @throws sham::format_error on formatting errors
     */
    template<typename... T>
    inline __attribute__((always_inline)) auto format_printf(
        std::string_view format, const T &...args) -> std::string {
        try {
            return fmt::sprintf(format, args...);
        } catch (const std::exception &e) {
            throw make_format_exception("printf", e.what(), std::string(format));
        }
    }

} // namespace sham

// Re-export all sham::format symbols into shambase for backwards compatibility
namespace shambase {

    using ::sham::format;
    using ::sham::format_printf;
    using ::sham::vformat;

} // namespace shambase
