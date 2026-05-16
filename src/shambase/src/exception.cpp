// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file exception.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/call_lambda.hpp"
#include "shambase/exception.hpp"
#include "shambase/format.hpp"
#include "shambase/stacktrace.hpp"
#include <string>

namespace shambase {

    std::string exception_format(SourceLocation loc) {
        return loc.format_multiline(fmt_callstack());
    }

    /// exception print callback func ptr
    void (*exception_print_callback)(std::string msg) = nullptr;

    void exception_gen_callback(std::string msg) {
        if (exception_print_callback != nullptr) {
            exception_print_callback(msg);
        }
    }

    void set_exception_gen_callback(exception_gen_callback_t callback) {
        exception_print_callback = callback;
    }

    exception_gen_callback_t get_exception_gen_callback() { return exception_print_callback; }

    std::format_error format_exception_builder(
        std::string_view function_call,
        std::string_view what,
        const std::string &fmt_string,
        std::source_location loc) {
        return make_except_with_loc<std::format_error>(
            fmt::format(
                "format failed:\n  function={}\n  what={}\n  fmt_string={}",
                function_call,
                what,
                fmt_string),
            SourceLocation(loc));
    }

    static shambase::call_lambda lambda_name([]() {
        set_format_exception_builder(format_exception_builder);
    });
} // namespace shambase
