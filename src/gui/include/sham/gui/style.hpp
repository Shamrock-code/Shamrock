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
 * @file style.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Design tokens of the GUI: colours and fixed bar heights.
 *
 */

#include "imgui.h"
#include "sham/gui/color_utils.hpp"

namespace sham::gui {

    // ============================================================================
    //  Design tokens
    // ============================================================================
    namespace theme {
        constexpr ImU32 APP_BG = rgba("#141517"), PANEL = rgba("#1b1c1f"), CANVAS = rgba("#17181b"),
                        DIVIDER = rgba("#2e3035"), BORDER = rgba("#34363c"),
                        BUTTON = rgba("#202125"), NODE = rgba("#232428"),
                        NODE_BORDER = rgba("#3a3c42"), CARD = rgba("#111214"),
                        DARK = rgba("#0b0c10"), GRID_DOT = rgba("#2b2d33"),
                        ROW_HL = rgba("#2a2b30");
        constexpr ImU32 TEXT = rgba("#e7e5df"), TEXT_2 = rgba("#c9c6be"), TEXT_3 = rgba("#a9a69e"),
                        ROW = rgba("#b3b0a8"), MUTED = rgba("#8e8b84"), DIM = rgba("#6f6d67"),
                        GUTTER = rgba("#5f5d58");
        constexpr ImU32 ACCENT = rgba("#e8a33d"), ACCENT_BG = rgba("#2a2418"),
                        ACCENT_TEXT = rgba("#f4e2c0"), ON_ACCENT = rgba("#1b1407"),
                        WARM_BG = rgba("#262015"), WARM_BORDER = rgba("#4a3b22"),
                        WARM_TEXT = rgba("#f1d7a8");
        constexpr ImU32 TEAL = rgba("#4fb3a9"), TEAL_TEXT = rgba("#8fb8b2"),
                        PILL_BG = rgba("#1a2624"), PILL_BORDER = rgba("#2f4d49"),
                        PILL_TEXT = rgba("#cfe9e4"), BLUE = rgba("#6f9be8"), GRAY = rgba("#8a8780");
        struct HeaderStyle {
            ImU32 bg, fg, chip;
        };
        constexpr HeaderStyle INPUT{rgba("#22403c"), rgba("#cfe9e4"), rgba("#2d5550")};
        constexpr HeaderStyle SOLVER{rgba("#43301f"), rgba("#f3dcc2"), rgba("#5c4029")};
    } // namespace theme

    // Heights of the top bar, status bar and pane headers (logical pixels).
    inline constexpr double TOP_H = 52.0, STATUS_H = 28.0, PANE_HDR = 36.0;

    /// Size settings and colour table of the dark theme (ImGui's own widgets: dock tabs,
    /// dividers, drop overlay, scrollbars).
    void setup_style();

} // namespace sham::gui
