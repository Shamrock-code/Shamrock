// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

    inline explicit SourceLocation(srcloc _loc = srcloc::current()) : loc(_loc) {}

    /**
     * @brief format the location in multiple lines
     *
     * @return std::string the formated location
     */
    std::string format_multiline();

    /**
     * @brief format the location in multiple lines with a given stacktrace
     *
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    std::string format_multiline(std::string stacktrace);

    /**
     * @brief format the location in a one liner
     *
     * @return std::string the formated location
     */
    std::string format_one_line();

    /**
     * @brief format the location in a one liner with the function name displayed
     *
     * @return std::string the formated location
     */
    std::string format_one_line_func();
};
