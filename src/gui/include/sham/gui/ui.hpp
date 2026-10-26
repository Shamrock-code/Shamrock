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
 * @file ui.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief UI scale state, logical <-> physical pixel conversion, mouse and cursor helpers, and text
 * measurement. Drawing in logical pixels goes through ScaledDrawList.hpp.
 *
 */

#include "imgui.h"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>

namespace sham::gui {

    inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

    /// UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); P() and S()
    /// convert them to physical pixels.
    struct UI {
        /// current scale (physical pixels per logical pixel)
        double scale = 1.0;

        /// main viewport position (scaling is done around it)
        ImVec2 origin{0, 0};

        /// range and step of the scale
        static constexpr double MIN = 0.5, MAX = 3.0, STEP = 0.1;

        /// logical point -> physical point
        ImVec2 P(ImVec2 p) const {
            const float s = float(scale);
            return ImVec2(origin.x + (p.x - origin.x) * s, origin.y + (p.y - origin.y) * s);
        }

        /// logical length -> physical length
        float S(double v) const { return float(v * scale); }

        /// mouse position in logical pixels
        ImVec2 mouse_pos() const {
            ImVec2 m = ImGui::GetIO().MousePos;
            return V(origin.x + (m.x - origin.x) / scale, origin.y + (m.y - origin.y) / scale);
        }

        /// mouse movement since the last frame in logical pixels
        ImVec2 mouse_delta() const {
            ImVec2 d = ImGui::GetIO().MouseDelta;
            return V(d.x / scale, d.y / scale);
        }
    };

    /// ImGui::SetCursorScreenPos taking a logical position
    inline void set_cursor(const UI &ui, ImVec2 p) { ImGui::SetCursorScreenPos(ui.P(p)); }

    /// ImGui::InvisibleButton taking a logical size (at least 1 physical pixel each way)
    inline bool invisible_button(
        const UI &ui, const char *id, ImVec2 size, ImGuiButtonFlags flags = 0) {
        return ImGui::InvisibleButton(
            id, V(std::max(size.x * ui.scale, 1.0), std::max(size.y * ui.scale, 1.0)), flags);
    }

    /// Key of the text_w() cache: the width depends on the scale through glyph rounding.
    struct TextKey {
        const ImFont *font;
        int size100;
        std::string s;
        int scale100;
        bool operator==(const TextKey &o) const {
            return font == o.font && size100 == o.size100 && s == o.s && scale100 == o.scale100;
        }
    };
    struct TextKeyHash {
        size_t operator()(const TextKey &k) const {
            return std::hash<std::string>()(k.s) ^ (std::hash<const void *>()(k.font) << 1)
                   ^ size_t(k.size100) * 31;
        }
    };
    inline std::unordered_map<TextKey, double, TextKeyHash> g_text_cache;

    /// Width in logical pixels of s drawn with font at the logical size size (cached).
    inline double text_w(const UI &ui, ImFont *font, double size, const std::string &s) {
        TextKey key{font, int(std::lround(size * 100)), s, int(std::lround(ui.scale * 100))};
        auto it = g_text_cache.find(key);
        if (it != g_text_cache.end())
            return it->second;
        ImGui::PushFont(font, float(size * ui.scale));
        double w = ImGui::CalcTextSize(s.c_str()).x / ui.scale;
        ImGui::PopFont();
        if (g_text_cache.size() > 4000)
            g_text_cache.clear();
        g_text_cache.emplace(std::move(key), w);
        return w;
    }

} // namespace sham::gui
