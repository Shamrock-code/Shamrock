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
 * @file format_exception.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declaration of the custom format exception builder system
 */

#include "sham/format/aliases.hpp"
#include <source_location>

namespace sham {

    /**
     * @brief Create a format error exception.
     *
     * Delegates to a custom builder if one is installed, otherwise returns
     * a default `fmt::format_error` constructed from `what`.
     *
     * @param function_call name of the function that triggered the error
     * @param what the error message from the underlying library
     * @param fmt_string the format string that caused the error
     * @param loc source location where the error occurred
     * @return sham::format_error the constructed exception
     */
    sham::format_error make_format_exception(
        std::string_view function_call,
        std::string_view what,
        const std::string &fmt_string,
        std::source_location loc = std::source_location::current());

    /**
     * @brief Type alias for a custom format exception builder function.
     *
     * Custom builders can enrich the error with metadata such as
     * source location or the original format string.
     */
    using format_except_builder_t = sham::format_error (*)(
        std::string_view, std::string_view, const std::string &, std::source_location);

    /**
     * @brief Install a custom builder for format exceptions.
     *
     * When set, all calls to `make_format_exception` will route through
     * the provided callback instead of using the default
     * `fmt::format_error(what)` constructor.
     *
     * @param callback the builder function pointer; passing `nullptr`
     *         resets to the default behavior
     */
    void set_format_exception_builder(format_except_builder_t callback);
} // namespace sham
