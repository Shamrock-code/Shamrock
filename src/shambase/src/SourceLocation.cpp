// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SourceLocation.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Source location utility
 * @date 2023-02-24
 */

#include "shambase/SourceLocation.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

std::string SourceLocation::format_multiline() {
    return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
-------------------------
)=",
        loc.file_name(),
        loc.line(),
        loc.column(),
        loc.function_name());
}

std::string SourceLocation::format_multiline(std::string stacktrace) {
    return fmt::format(
R"=(
---- Source Location ----
{}:{}:{}
call = {}
stacktrace : 
{}
-------------------------
)=",
        loc.file_name(),
        loc.line(),
        loc.column(),
        loc.function_name(),
        stacktrace);
}

std::string SourceLocation::format_one_line() {
    return fmt::format("{}:{}:{}", loc.file_name(), loc.line(), loc.column());
}

std::string SourceLocation::format_one_line_func() {
    return fmt::format(
        "{} ({}:{}:{})", loc.function_name(), loc.file_name(), loc.line(), loc.column());
}