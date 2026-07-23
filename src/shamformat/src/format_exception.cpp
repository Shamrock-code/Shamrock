// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file format_exception.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "sham/format/format.hpp"

namespace sham {

    /// Internal function ptr handle.
    /// Must be not static to point to the same space across multiple shared libraries
    format_except_builder_t internal_func_ptr_make_format_exception = nullptr;

    sham::format_error make_format_exception(
        std::string_view function_call,
        std::string_view what,
        const std::string &fmt_string,
        std::source_location loc) {
        if (internal_func_ptr_make_format_exception != nullptr) {
            return internal_func_ptr_make_format_exception(function_call, what, fmt_string, loc);
        } else {
            return sham::format_error(std::string(what));
        }
    }

    void set_format_exception_builder(format_except_builder_t callback) {
        internal_func_ptr_make_format_exception = callback;
    }

    format_except_builder_t get_format_exception_builder() noexcept {
        return internal_func_ptr_make_format_exception;
    }

} // namespace sham
