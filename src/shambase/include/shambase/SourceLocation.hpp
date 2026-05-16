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
 * @file SourceLocation.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Source location utility
 * @date 2023-02-24
 */

#include <source_location>
#include <string>

namespace shambase {
    /**
     * @brief format the location in multiple lines
     *
     * @return std::string the formated location
     */
    std::string format_multiline(std::source_location loc);

    /**
     * @brief format the location in multiple lines with a given stacktrace
     *
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    std::string format_multiline(std::source_location loc, const std::string &stacktrace);

    /**
     * @brief format the location in a one liner
     *
     * @return std::string the formated location
     */
    std::string format_one_line(std::source_location loc);

    /**
     * @brief format the location in a one liner with the function name displayed
     *
     * @return std::string the formated location
     */
    std::string format_one_line_func(std::source_location loc);
} // namespace shambase

/**
 * @brief provide information about the source location
 *
 * Example :
 * \code{.cpp}
 * SourceLocation loc = SourceLocation{};
 * \endcode
 */
struct SourceLocation {

    using srcloc = std::source_location;

    srcloc loc;

    inline constexpr explicit SourceLocation(srcloc _loc = srcloc::current()) noexcept
        : loc(_loc) {}

    /**
     * @brief format the location in multiple lines
     *
     * @return std::string the formated location
     */
    inline std::string format_multiline() const { return shambase::format_multiline(loc); }

    /**
     * @brief format the location in multiple lines with a given stacktrace
     *
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    inline std::string format_multiline(const std::string &stacktrace) const {
        return shambase::format_multiline(loc, stacktrace);
    }

    /**
     * @brief format the location in a one liner
     *
     * @return std::string the formated location
     */
    inline std::string format_one_line() const { return shambase::format_one_line(loc); }

    /**
     * @brief format the location in a one liner with the function name displayed
     *
     * @return std::string the formated location
     */
    inline std::string format_one_line_func() const { return shambase::format_one_line_func(loc); }
};
