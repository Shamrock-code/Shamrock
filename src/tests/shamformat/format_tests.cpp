// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "sham/format/format.hpp"
#include "sham/format/human_readable.hpp"
#include "shamtest/shamtest.hpp"
#include <string_view>

namespace {
    void throwing_format_std() {
        std::string fmt = "{"; // runtime format string
        int value       = 42;

        auto s = sham::vformat(std::string_view{fmt}, fmt::make_format_args(value));

        // just to trap the result and avoid optimizations
        std::cout << s << '\n';
    }

    void throwing_format_fmt() {
        std::string fmt = "{"; // runtime format string
        int value       = 42;

        auto s = sham::vformat(fmt::string_view{fmt}, fmt::make_format_args(value));

        // just to trap the result and avoid optimizations
        std::cout << s << '\n';
    }
} // namespace

NEW_TEST(Unittest, "shamformat/format(throwing)", 1) {
    REQUIRE_EXCEPTION_THROW(throwing_format_std(), fmt::format_error);
    REQUIRE_EXCEPTION_THROW(throwing_format_fmt(), fmt::format_error);
}

NEW_TEST(Unittest, "shamformat/human_readable", 1) {
    using sham::human_readable_t;
    using sham::to_human_readable;

    // Zero: no prefix
    {
        auto hr = to_human_readable(0.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 0.0, 1e-15);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // Too small (below nano): clamped to nano
    {
        auto hr = to_human_readable(1e-10);
        REQUIRE_FLOAT_EQUAL(hr.value, 0.1, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "n");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e-9, 1e-15);
    }

    // Too large (above yotta): clamped to yotta
    {
        auto hr = to_human_readable(1e26);
        REQUIRE_FLOAT_EQUAL(hr.value, 100.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e24, 1e20);
    }

    // Exactly at boundaries
    {
        auto hr = to_human_readable(1e-9);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "n");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e-9, 1e-15);
    }

    {
        auto hr = to_human_readable(1e24);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1e24, 1e20);
    }

    // Common SI prefixes
    {
        auto hr = to_human_readable(1e3);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    {
        auto hr = to_human_readable(1e6);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "M");
    }

    {
        auto hr = to_human_readable(1e9);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "G");
    }

    {
        auto hr = to_human_readable(1e12);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "T");
    }

    // No prefix needed (value in [1, 1000))
    {
        auto hr = to_human_readable(500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 500.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // Negative values
    {
        auto hr = to_human_readable(-1e3);
        REQUIRE_FLOAT_EQUAL(hr.value, -1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    // Boundary: just below kilo (999.999)
    {
        auto hr = to_human_readable(999.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 999.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
    }

    // Between prefixes: 2500 -> 2.5 k
    {
        auto hr = to_human_readable(2500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 2.5, 1e-3);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    // Peta/E zetta: large non-standard values
    {
        auto hr = to_human_readable(5e15);
        REQUIRE_FLOAT_EQUAL(hr.value, 5.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "P");
    }
}

NEW_TEST(Unittest, "shamformat/human_readable", 1) {
    using sham::to_human_readable;

    // Zero: no prefix (always)
    {
        auto hr = to_human_readable<false>(0.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 0.0, 1e-15);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // Values below 1: clamped to "" (1.0) since nano/micro/milli are excluded
    {
        auto hr = to_human_readable<false>(1e-10);
        REQUIRE_FLOAT_EQUAL(hr.value, 1e-10, 1e-15);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // 1.0 stays as ""
    {
        auto hr = to_human_readable<false>(1.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
        REQUIRE_FLOAT_EQUAL(hr.ratio, 1.0, 1e-15);
    }

    // 500 stays as ""
    {
        auto hr = to_human_readable<false>(500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 500.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
    }

    // 999.999 stays as "" (just below kilo)
    {
        auto hr = to_human_readable<false>(999.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 999.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "");
    }

    // 1000 -> 1.0 k
    {
        auto hr = to_human_readable<false>(1000.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "k");
    }

    // 1e6 -> 1.0 M
    {
        auto hr = to_human_readable<false>(1e6);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "M");
    }

    // 1e24 -> 1.0 Y
    {
        auto hr = to_human_readable<false>(1e24);
        REQUIRE_FLOAT_EQUAL(hr.value, 1.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
    }

    // Too large: clamped to yotta
    {
        auto hr = to_human_readable<false>(1e26);
        REQUIRE_FLOAT_EQUAL(hr.value, 100.0, 1e-6);
        REQUIRE_EQUAL(hr.prefix, "Y");
    }

    // Between prefixes: 2500 -> 2.5 k
    {
        auto hr = to_human_readable<false>(2500.0);
        REQUIRE_FLOAT_EQUAL(hr.value, 2.5, 1e-3);
        REQUIRE_EQUAL(hr.prefix, "k");
    }
}
