// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
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

std::string shambase::format_multiline(std::source_location loc) {
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

std::string shambase::format_multiline(std::source_location loc, const std::string &stacktrace) {
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

std::string shambase::format_one_line(std::source_location loc) {
    return fmt::format("{}:{}:{}", loc.file_name(), loc.line(), loc.column());
}

std::string shambase::format_one_line_func(std::source_location loc) {
    return fmt::format(
        "{} ({}:{}:{})", loc.function_name(), loc.file_name(), loc.line(), loc.column());
}
