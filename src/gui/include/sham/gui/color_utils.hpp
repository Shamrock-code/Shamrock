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
 * @file color_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compile-time colour helpers: packed ImU32 colours from RGB components or "#rrggbb"
 * strings.
 *
 */

#include "imgui.h"

namespace sham::gui {

    /// Value of one hexadecimal digit ('0'-'9', 'a'-'f' or 'A'-'F').
    constexpr int hexv(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

    /// Pack 8-bit components into an ImU32 (ImGui's IM_COL32 layout).
    constexpr ImU32 rgb_u32(int r, int g, int b, int a = 255) {
        return (ImU32(a) << 24) | (ImU32(b) << 16) | (ImU32(g) << 8) | ImU32(r);
    }

    /// Colour from a "#rrggbb" string, with alpha a in [0, 1].
    constexpr ImU32 rgba(const char *h, double a = 1.0) {
        return rgb_u32(
            hexv(h[1]) * 16 + hexv(h[2]),
            hexv(h[3]) * 16 + hexv(h[4]),
            hexv(h[5]) * 16 + hexv(h[6]),
            int(a * 255 + 0.5));
    }

} // namespace sham::gui
