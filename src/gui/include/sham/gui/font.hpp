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
 * @file font.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief IBM Plex fonts used by the GUI (Sans for UI text, Mono for values and code).
 *
 */

#include "imgui.h"
#include <filesystem>
#include <string>

namespace sham::gui {

    // Non-owning: the fonts belong to ImGui's font atlas and are freed by ImGui::DestroyContext().
    struct Fonts {
        ImFont *sans     = nullptr;
        ImFont *medium   = nullptr;
        ImFont *semibold = nullptr;
        ImFont *mono     = nullptr;
    };

    inline auto load_fonts(const std::filesystem::path &font_folder) -> Fonts {
        ImGuiIO &io = ImGui::GetIO();
        auto load   = [&](const char *name, bool merge = false) {
            ImFontConfig cfg;
            cfg.MergeMode    = merge;
            std::string path = (font_folder / (std::string(name) + ".ttf")).string();
            ImFont *f        = io.Fonts->AddFontFromFileTTF(path.c_str(), 13.0f, &cfg);
            IM_ASSERT(f && "font not found: run from the project folder or pass --assets");
            return f;
        };
        Fonts fonts;
        fonts.sans     = load("IBMPlexSans-Regular");
        fonts.medium   = load("IBMPlexSans-Medium");
        fonts.semibold = load("IBMPlexSans-SemiBold");
        fonts.mono     = load("IBMPlexMono-Regular");
        load("IBMPlexSans-Regular", true); // Plex Mono has no Greek (rho): borrow it from Sans
        return fonts;
    }

} // namespace sham::gui
