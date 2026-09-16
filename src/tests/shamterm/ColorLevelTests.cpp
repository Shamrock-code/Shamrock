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
#include <cstdint>
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

    // Truecolor (24-bit RGB, \x1b[38;2;r;g;bm): print a rainbow gradient bar exercising the
    // full RGB range.
    sham::term::enable_colors();
    sham::term::set_color_level(sham::term::ColorLevel::TrueColor);

    shambase::println("truecolor:");

    constexpr int width = 77;
    for (int col = 0; col < width; col++) {
        int r = 255 - (col * 255 / (width - 1));
        int g = col * 510 / (width - 1);
        int b = col * 255 / (width - 1);
        if (g > 255) {
            g = 510 - g;
        }
        shambase::print(
            sham::term::colors_24b::background(
                static_cast<std::uint8_t>(r),
                static_cast<std::uint8_t>(g),
                static_cast<std::uint8_t>(b)));
        shambase::print(
            sham::term::colors_24b::foreground(
                static_cast<std::uint8_t>(255 - r),
                static_cast<std::uint8_t>(255 - g),
                static_cast<std::uint8_t>(255 - b)));
        shambase::print((col % 2 == 0) ? "/" : "\\");
        shambase::print(sham::term::style::reset());
    }
    shambase::println("");

    std::string fg = sham::term::colors_24b::foreground(12, 34, 56);
    REQUIRE_EQUAL_NAMED("truecolor foreground escape is well formed", fg, "\x1b[38;2;12;34;56m");

    std::string bg = sham::term::colors_24b::background(12, 34, 56);
    REQUIRE_EQUAL_NAMED("truecolor background escape is well formed", bg, "\x1b[48;2;12;34;56m");

    // 256-color palette (16 ANSI + 216-color cube + 24 grayscale, \x1b[38;5;Nm)
    sham::term::set_color_level(sham::term::ColorLevel::ANSI256);
    REQUIRE_EQUAL_NAMED(
        "truecolor escape is empty below ColorLevel::TrueColor",
        sham::term::colors_24b::foreground(12, 34, 56),
        "");

    shambase::println("ansi256:");

    // 216-color cube (indices 16-231), each cell paired with a contrasting foreground index.
    for (int i = 16; i <= 231; i++) {
        shambase::print(sham::term::colors_256::background(static_cast<std::uint8_t>(i)));
        shambase::print(sham::term::colors_256::foreground(static_cast<std::uint8_t>(255 - i)));
        shambase::print((i % 2 == 0) ? "/" : "\\");
        shambase::print(sham::term::style::reset());
    }
    shambase::println("");

    // 24-step grayscale ramp (indices 232-255).
    for (int i = 232; i <= 255; i++) {
        shambase::print(sham::term::colors_256::background(static_cast<std::uint8_t>(i)));
        shambase::print(sham::term::colors_256::foreground(static_cast<std::uint8_t>(255 - i)));
        shambase::print((i % 2 == 0) ? "/" : "\\");
        shambase::print(sham::term::style::reset());
    }
    shambase::println("");

    std::string fg256 = sham::term::colors_256::foreground(196);
    REQUIRE_EQUAL_NAMED("256-color foreground escape is well formed", fg256, "\x1b[38;5;196m");

    std::string bg256 = sham::term::colors_256::background(196);
    REQUIRE_EQUAL_NAMED("256-color background escape is well formed", bg256, "\x1b[48;5;196m");

    // ANSI/16 colors (basic SGR codes)
    sham::term::set_color_level(sham::term::ColorLevel::Basic);
    REQUIRE_EQUAL_NAMED(
        "256-color escape is empty below ColorLevel::ANSI256",
        sham::term::colors_256::foreground(196),
        "");

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
