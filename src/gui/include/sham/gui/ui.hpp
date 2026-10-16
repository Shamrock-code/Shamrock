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
 * @brief UI scale layer: logical-pixel drawing (SDL), mouse and cursor helpers, and text
 * measurement.
 *
 */

#include "imgui.h"
#include "sham/gui/font.hpp"
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace sham::gui {

    /// fonts of the GUI, set by load_fonts() in main()
    inline Fonts g_fonts;

    // ============================================================================
    //  Small drawing helpers
    // ============================================================================
    inline ImVec2 V(double x, double y) { return ImVec2(float(x), float(y)); }

    /// Global UI scale. The app is laid out in *logical* pixels (the mockup's 1x sizes); these
    /// helpers convert to physical pixels when drawing, placing widgets and reading the mouse.
    struct UI {
        /// current scale (physical pixels per logical pixel)
        static inline double scale = 1.0;

        /// main viewport position (scaling is done around it)
        static inline ImVec2 origin{0, 0};

        /// range and step of the scale
        static constexpr double MIN = 0.5, MAX = 3.0, STEP = 0.1;
    };

    /// logical point -> physical point
    inline ImVec2 P(ImVec2 p) {
        const float s = float(UI::scale);
        return ImVec2(
            UI::origin.x + (p.x - UI::origin.x) * s, UI::origin.y + (p.y - UI::origin.y) * s);
    }

    /// logical length -> physical length
    inline float S(double v) { return float(v * UI::scale); }

    /// mouse position in logical pixels
    inline ImVec2 mouse_pos() {
        ImVec2 m = ImGui::GetIO().MousePos;
        return V(
            UI::origin.x + (m.x - UI::origin.x) / UI::scale,
            UI::origin.y + (m.y - UI::origin.y) / UI::scale);
    }

    /// mouse movement since the last frame in logical pixels
    inline ImVec2 mouse_delta() {
        ImVec2 d = ImGui::GetIO().MouseDelta;
        return V(d.x / UI::scale, d.y / UI::scale);
    }

    /// ImGui::SetCursorScreenPos taking a logical position
    inline void set_cursor(ImVec2 p) { ImGui::SetCursorScreenPos(P(p)); }

    /// ImGui::InvisibleButton taking a logical size (at least 1 physical pixel each way)
    inline bool invisible_button(const char *id, ImVec2 size, ImGuiButtonFlags flags = 0) {
        return ImGui::InvisibleButton(
            id, V(std::max(size.x * UI::scale, 1.0), std::max(size.y * UI::scale, 1.0)), flags);
    }

    /// Same calls as ImDrawList, but taking logical coordinates and sizes.
    struct SDL {
        ImDrawList *d = nullptr;
        void AddRectFilled(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0) {
            d->AddRectFilled(P(a), P(b), c, S(r), f);
        }
        void AddRect(ImVec2 a, ImVec2 b, ImU32 c, float r = 0, ImDrawFlags f = 0, float th = 1) {
            d->AddRect(P(a), P(b), c, S(r), f, S(th));
        }
        void AddRectFilledMultiColor(ImVec2 a, ImVec2 b, ImU32 c1, ImU32 c2, ImU32 c3, ImU32 c4) {
            d->AddRectFilledMultiColor(P(a), P(b), c1, c2, c3, c4);
        }
        void AddLine(ImVec2 a, ImVec2 b, ImU32 c, float th = 1) {
            d->AddLine(P(a), P(b), c, S(th));
        }
        void AddCircle(ImVec2 c, float r, ImU32 col, int seg = 0, float th = 1) {
            d->AddCircle(P(c), S(r), col, seg, S(th));
        }
        void AddCircleFilled(ImVec2 c, float r, ImU32 col, int seg = 0) {
            d->AddCircleFilled(P(c), S(r), col, seg);
        }
        void AddTriangle(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col, float th = 1) {
            d->AddTriangle(P(a), P(b), P(c), col, S(th));
        }
        void AddTriangleFilled(ImVec2 a, ImVec2 b, ImVec2 c, ImU32 col) {
            d->AddTriangleFilled(P(a), P(b), P(c), col);
        }
        void AddConvexPolyFilled(const ImVec2 *pts, int n, ImU32 col) {
            auto q = scaled(pts, n);
            d->AddConvexPolyFilled(q.data(), n, col);
        }
        void AddPolyline(const ImVec2 *pts, int n, ImU32 col, ImDrawFlags f, float th) {
            auto q = scaled(pts, n);
            d->AddPolyline(q.data(), n, col, f, S(th));
        }
        void AddBezierCubic(
            ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e, ImU32 col, float th, int seg = 0) {
            d->AddBezierCubic(P(a), P(b), P(c), P(e), col, S(th), seg);
        }
        void AddText(ImFont *f, float size, ImVec2 pos, ImU32 col, const char *s) {
            d->AddText(f, S(size), P(pos), col, s);
        }
        void AddImage(ImTextureRef t, ImVec2 a, ImVec2 b) { d->AddImage(t, P(a), P(b)); }
        void AddImageRounded(
            ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 uv0, ImVec2 uv1, ImU32 col, float r) {
            d->AddImageRounded(t, P(a), P(b), uv0, uv1, col, S(r));
        }
        void AddImageQuad(ImTextureRef t, ImVec2 a, ImVec2 b, ImVec2 c, ImVec2 e) {
            d->AddImageQuad(t, P(a), P(b), P(c), P(e));
        }
        void PushClipRect(ImVec2 a, ImVec2 b, bool intersect = false) {
            d->PushClipRect(P(a), P(b), intersect);
        }
        void PopClipRect() { d->PopClipRect(); }

        private:
        static std::vector<ImVec2> scaled(const ImVec2 *pts, int n) {
            std::vector<ImVec2> q(n);
            for (int i = 0; i < n; ++i)
                q[i] = P(pts[i]);
            return q;
        }
    };
    inline SDL g_sdl[16];
    inline int g_sdl_next = 0; // reset at the start of each frame

    /// One wrapper per window drawn in the frame (main window + panes).
    inline SDL *window_draw_list() {
        SDL *s = &g_sdl[g_sdl_next++ % 16];
        s->d   = ImGui::GetWindowDrawList();
        return s;
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
    inline double text_w(ImFont *font, double size, const std::string &s) {
        TextKey key{font, int(std::lround(size * 100)), s, int(std::lround(UI::scale * 100))};
        auto it = g_text_cache.find(key);
        if (it != g_text_cache.end())
            return it->second;
        ImGui::PushFont(font, float(size * UI::scale));
        double w = ImGui::CalcTextSize(s.c_str()).x / UI::scale;
        ImGui::PopFont();
        if (g_text_cache.size() > 4000)
            g_text_cache.clear();
        g_text_cache.emplace(std::move(key), w);
        return w;
    }

    /// Draw s with its top-left corner at (x, y) (logical pixels).
    inline void draw_text(
        SDL *dl, ImFont *font, double size, double x, double y, ImU32 col, const std::string &s) {
        dl->AddText(font, float(size), V(x, y), col, s.c_str());
    }

    /// Draw s starting at x, vertically centred on cy (logical pixels).
    inline void draw_text_vc(
        SDL *dl, ImFont *font, double size, double x, double cy, ImU32 col, const std::string &s) {
        dl->AddText(font, float(size), V(x, cy - size * 0.66), col, s.c_str());
    }

} // namespace sham::gui
