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

#include "shambase/source_location.hpp"
#include <optional>
#include <string>

/**
 * @brief provide information about the source location
 *
 * Exemple :
 * \code{.cpp}
 * SourceLocation loc = SourceLocation{};
 * \endcode
 */
struct SourceLocation {

    using srcloc = shambase::cxxstd::source_location;

    srcloc loc;
    std::optional<std::string> message;

    inline explicit SourceLocation(
        std::optional<std::string> _message = std::nullopt, srcloc _loc = srcloc::current())
        : loc(_loc), message(_message) {}

    /**
     * @brief format the location in multiple lines
     *
     * @return std::string the formated location
     */
    std::string format_multiline() const;

    /**
     * @brief format the location in multiple lines with a given stacktrace
     *
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    std::string format_multiline(std::string stacktrace) const;

    /**
     * @brief format the location in a one liner
     *
     * @return std::string the formated location
     */
    std::string format_one_line() const;

    /**
     * @brief format the location in a one liner with the function name displayed
     *
     * @return std::string the formated location
     */
    std::string format_one_line_func() const;
};
