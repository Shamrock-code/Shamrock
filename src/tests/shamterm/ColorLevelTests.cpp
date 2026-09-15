// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/print.hpp"
#include "sham/term/color.hpp"
#include "shamtest/shamtest.hpp"
#include <string>

namespace {

    /// Saves/restores sham::term's color enable state and level, since both are process-global
    /// singletons shared with every other test in this binary.
    struct ColorStateGuard {
        bool enabled                 = sham::term::are_colors_enabled();
        sham::term::ColorLevel level = sham::term::color_level();

        ~ColorStateGuard() {
            if (enabled) {
                sham::term::enable_colors();
            } else {
                sham::term::disable_colors();
            }
            sham::term::set_color_level(level);
        }
    };

} // namespace

NEW_TEST(Unittest, "shamterm/color", 1) {
    ColorStateGuard guard{};

    // ANSI/16 colors (basic SGR codes)
    sham::term::enable_colors();
    sham::term::set_color_level(sham::term::ColorLevel::Basic);

    shambase::println("basic:");

    struct NamedColor {
        const char *name;
        const char *(*escape)();
    };
    const NamedColor colors[] = {
        {.name = "black", .escape = sham::term::colors_8b::black},
        {.name = "red", .escape = sham::term::colors_8b::red},
        {.name = "green", .escape = sham::term::colors_8b::green},
        {.name = "yellow", .escape = sham::term::colors_8b::yellow},
        {.name = "blue", .escape = sham::term::colors_8b::blue},
        {.name = "magenta", .escape = sham::term::colors_8b::magenta},
        {.name = "cyan", .escape = sham::term::colors_8b::cyan},
        {.name = "white", .escape = sham::term::colors_8b::white},
    };

    for (auto &c : colors) {
        shambase::print(c.escape());
        shambase::print(std::string(" ") + c.name + " ");
        shambase::print(sham::term::style::reset());
    }
    shambase::println("");

    for (auto &c : colors) {
        REQUIRE_EQUAL_NAMED(
            std::string(c.name) + " escape is non-empty", std::string(c.escape()).empty(), false);
    }

    sham::term::disable_colors();
    for (auto &c : colors) {
        REQUIRE_EQUAL_NAMED(
            std::string(c.name) + " escape is empty when colors are disabled",
            std::string(c.escape()),
            "");
    }
}
