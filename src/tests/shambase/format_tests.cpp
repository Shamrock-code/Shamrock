// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/format.hpp"
#include "shamtest/shamtest.hpp"
#include <string_view>
#include <format>

namespace {
    void throwing_format() {
        std::string fmt = "{"; // runtime format string
        int value       = 42;

        auto s = shambase::vformat(std::string_view{fmt}, fmt::make_format_args(value));

        // just to trap the result and avoid optimizations
        std::cout << s << '\n';
    }
} // namespace

TestStart(Unittest, "shambase/format", test_exception_throw, 1) {
    REQUIRE_EXCEPTION_THROW(throwing_format(), std::format_error);
}
