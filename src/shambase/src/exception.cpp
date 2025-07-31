// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambase/exception.hpp"
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

    void set_exception_gen_callback(void (*callback)(std::string msg)) {
        exception_print_callback = callback;
    }

} // namespace shambase
