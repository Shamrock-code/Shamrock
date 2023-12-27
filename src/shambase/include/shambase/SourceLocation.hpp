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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Source location utility
 * @date 2023-02-24
 */

#include "shambase/aliases_int.hpp"
#include "shambase/source_location.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

/**
 * @brief provide information about the source location
 * 
 * Exemple :
 * \code{.cpp}
 * SourceLocation loc = SourceLocation{};
 * \endcode
 */
struct SourceLocation {

    shambase::cxxstd::source_location loc;

    inline explicit SourceLocation(

        shambase::cxxstd::source_location _loc = shambase::cxxstd::source_location::current()

        ) : 
            loc(_loc) {}

    /**
     * @brief format the location in multiple lines
     * 
     * @return std::string the formated location
     */
    std::string format_multiline(){
        return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
-------------------------
)="
            , loc.file_name(), loc.line(), loc.column(), loc.function_name());
    }

    /**
     * @brief format the location in multiple lines with a given stacktrace
     * 
     * @param stacktrace the stacktrace to add to the location
     * @return std::string the formated location
     */
    std::string format_multiline(std::string stacktrace){
        return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
stacktrace : 
{}
-------------------------
)="
            ,loc.file_name(), loc.line(), loc.column(), loc.function_name(),stacktrace);
    }

    /**
     * @brief format the location in a one liner
     * 
     * @return std::string the formated location
     */
    std::string format_one_line(){
        return fmt::format("{}:{}:{}", loc.file_name(), loc.line(), loc.column());
    }

    /**
     * @brief format the location in a one liner with the function name displayed
     * 
     * @return std::string the formated location
     */
    std::string format_one_line_func(){
        return fmt::format("{} ({}:{}:{})", loc.function_name(), loc.file_name(), loc.line(), loc.column());
    }
};