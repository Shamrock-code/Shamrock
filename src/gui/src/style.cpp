// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file style.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Dark theme of shamrock_gui applied to ImGui's style, and its UI scaling.
 *
 */

#include "sham/gui/style.hpp"
#include "imgui.h"
#include <algorithm>
#include <utility>

namespace sham::gui {

    // unscaled style from setup_style(); apply_scale() starts from it
    static ImGuiStyle base_style;

    void setup_style() {
        ImGuiStyle &style                       = ImGui::GetStyle();
        style.FramePadding                      = ImVec2(12, 6); // dock tab height = font + 2 * 6
        style.TabRounding                       = 0;
        style.TabBarBorderSize                  = 1;
        style.TabBarOverlineSize                = 2;
        style.TabBorderSize                     = 0;
        style.DockingSeparatorSize              = 1;
        style.WindowMenuButtonPosition          = ImGuiDir_None;
        style.TabCloseButtonMinWidthSelected    = 0; // close cross only when hovered
        style.TabCloseButtonMinWidthUnselected  = 0;
        style.WindowPadding                     = ImVec2(0, 0);
        style.WindowBorderSize                  = 0;
        style.ChildBorderSize                   = 0;
        style.WindowRounding                    = 0;
        style.ScrollbarSize                     = 10;
        style.ScrollbarRounding                 = 4;
        style.ItemSpacing                       = ImVec2(0, 0);
        const std::pair<ImGuiCol, ImU32> cols[] = {
            {ImGuiCol_WindowBg, theme::APP_BG},
            {ImGuiCol_ChildBg, theme::CANVAS},
            {ImGuiCol_ScrollbarBg, theme::CANVAS},
            {ImGuiCol_ScrollbarGrab, theme::BORDER},
            {ImGuiCol_ScrollbarGrabHovered, theme::NODE_BORDER},
            {ImGuiCol_ScrollbarGrabActive, theme::MUTED},
            {ImGuiCol_Text, theme::TEXT},
            {ImGuiCol_PopupBg, theme::PANEL},
            {ImGuiCol_Border, theme::BORDER},
            {ImGuiCol_TextSelectedBg, rgba("#e8a33d", 0.25)},
            // docking: tab bars, drop preview, dividers
            {ImGuiCol_TitleBg, theme::PANEL},
            {ImGuiCol_TitleBgActive, theme::PANEL},
            {ImGuiCol_TitleBgCollapsed, theme::PANEL},
            {ImGuiCol_Tab, theme::PANEL},
            {ImGuiCol_TabHovered, theme::BUTTON},
            {ImGuiCol_TabSelected, theme::CANVAS},
            {ImGuiCol_TabSelectedOverline, theme::ACCENT},
            {ImGuiCol_TabDimmed, theme::PANEL},
            {ImGuiCol_TabDimmedSelected, theme::CANVAS},
            {ImGuiCol_TabDimmedSelectedOverline, rgba("#e8a33d", 0.35)},
            {ImGuiCol_DockingPreview, rgba("#e8a33d", 0.30)},
            {ImGuiCol_DockingEmptyBg, theme::CANVAS},
            {ImGuiCol_Separator, theme::DIVIDER},
            {ImGuiCol_SeparatorHovered, rgba("#e8a33d", 0.6)},
            {ImGuiCol_SeparatorActive, theme::ACCENT},
            {ImGuiCol_Button, 0},
            {ImGuiCol_ButtonHovered, theme::ROW_HL},
            {ImGuiCol_ButtonActive, theme::ACCENT_BG},
            {ImGuiCol_FrameBg, theme::BUTTON},
        };
        for (auto &[k, v] : cols)
            style.Colors[k] = ImGui::ColorConvertU32ToFloat4(v);
        base_style = style;
    }

    void apply_scale(double s) {
        ImGuiStyle &style = ImGui::GetStyle();
        style             = base_style;
        style.ScaleAllSizes(float(s));
        style.FontSizeBase         = float(13 * s);
        style.DockingSeparatorSize = std::max(1.0f, float(s));
    }

} // namespace sham::gui
