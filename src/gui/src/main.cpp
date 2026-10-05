// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file main.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shamrock control GUI: for now a Dear ImGui (docking branch) frame loop showing an empty
 * dock area that fills the window, with headless modes (deterministic 60 fps clock for
 * --screenshot and --bench).
 *
 * Interactive runs remember the dock arrangement in shamrock_gui_layout.ini.
 *
 * Usage:
 *
 *     ./shamrock_gui                        interactive
 *     ./shamrock_gui --screenshot shot.png  render 45 frames (or --frames N), save PNG, exit
 *     ./shamrock_gui --bench 300            print per-frame CPU timings as JSON
 *
 */

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "sham/gui/FrameTimings.hpp"
#include "sham/gui/GuiClock.hpp"
#include "sham/gui/screenshot.hpp"
#include <GLFW/glfw3.h>
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

namespace sham::gui {

    /// Build one frame: a full-screen host window holding the dock area.
    void gui() {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove
                                 | ImGuiWindowFlags_NoSavedSettings
                                 | ImGuiWindowFlags_NoBringToFrontOnFocus
                                 | ImGuiWindowFlags_NoScrollWithMouse;
        // no padding or border, so the dock area covers the whole window
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGui::Begin("##shamrock_main", nullptr, flags);
        ImGui::PopStyleVar(2);
        ImGui::DockSpace(ImGui::GetID("##body_dockspace"), ImVec2(0, 0));
        ImGui::End();
    }

    /// Command-line options of shamrock_gui: the flags as given, and what follows from them.
    struct CliArgs {
        /// --screenshot PATH: save a PNG of the window before exiting (an empty PATH is ignored)
        std::optional<std::string> screenshot = std::nullopt;

        /// --frames N: frames rendered before saving the screenshot
        std::optional<int> frames = std::nullopt;

        /// --bench N: render N frames (after 30 warm-up frames) and print per-frame timings
        std::optional<int> bench = std::nullopt;

        /// -h / --help
        std::optional<bool> is_help = std::nullopt;

        /// first unrecognised argument (parsing stops there)
        std::optional<std::string> unknown_arg = std::nullopt;

        /// frames rendered and discarded before the --bench timings are kept
        static constexpr int bench_warmup_frames = 30;

        /// false for headless runs (--screenshot, --bench): deterministic clock, no vsync, no .ini
        bool interactive_mode() const { return !screenshot && !bench; }

        /// frames rendered before exiting: --bench N + 30 warm-up frames, else --frames (default
        /// 45) with --screenshot, empty for an interactive run
        std::optional<int> frames_before_exit() const {
            if (bench)
                return *bench + bench_warmup_frames;
            if (screenshot)
                return frames.value_or(45);
            return std::nullopt;
        }

        /// set when main must print the usage and return right away
        std::optional<int> exit_code() const {
            if (is_help.value_or(false))
                return 0;
            if (unknown_arg)
                return 1;
            return std::nullopt;
        }
    };

    /// Parse argv into CliArgs; stops at -h / --help or at the first unknown option.
    static CliArgs parse_cli(int argc, char **argv) {
        CliArgs cli;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next     = [&]() {
                return i + 1 < argc ? std::string(argv[++i]) : std::string();
            };
            if (a == "--screenshot") {
                if (std::string path = next(); !path.empty())
                    cli.screenshot = path;
            } else if (a == "--frames") {
                cli.frames = std::stoi(next());
            } else if (a == "--bench") {
                if (int n = std::stoi(next()); n > 0)
                    cli.bench = n;
            } else if (a == "-h" || a == "--help") {
                cli.is_help = true;
                break;
            } else {
                cli.unknown_arg = a;
                break;
            }
        }
        return cli;
    }

} // namespace sham::gui

int main(int argc, char **argv) {
    using namespace sham::gui;
    const CliArgs cli = parse_cli(argc, argv);
    if (std::optional<int> code = cli.exit_code()) {
        std::printf("usage: %s [--screenshot out.png] [--frames N] [--bench N]\n", argv[0]);
        return *code;
    }

    if (!glfwInit()) {
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow *window = glfwCreateWindow(1440, 960, "Shamrock", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(cli.interactive_mode() ? 1 : 0);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // interactive runs remember the arrangement; screenshots always start from scratch
    io.IniFilename = cli.interactive_mode() ? "shamrock_gui_layout.ini" : nullptr;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    GuiClock gui_clock(!cli.interactive_mode());
    FrameTimings timings; // only filled with --bench

    int fbw = 0, fbh = 0;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (cli.bench)
            timings.begin_frame();
        // no data-update step yet, so "update" reads about 0 ms; the --bench JSON keeps the key
        // so its format stays stable once the update step lands
        if (cli.bench)
            timings.mark_update();
        gui();
        if (cli.bench)
            timings.mark_ui();
        gui_clock.end_frame();
        const bool want_exit
            = cli.frames_before_exit() && gui_clock.frame_counter >= *cli.frames_before_exit();
        // temporary: something moving to check --screenshot, removed with the real panes
        {
            const double t = gui_clock.now();
            const ImVec2 c(360 + 200 * float(std::cos(t)), 240 + 120 * float(std::sin(2 * t)));
            ImGui::GetForegroundDrawList()->AddRectFilled(
                ImVec2(c.x - 20, c.y - 20),
                ImVec2(c.x + 20, c.y + 20),
                IM_COL32(232, 163, 61, 255));
        }
        ImGui::Render();
        glfwGetFramebufferSize(window, &fbw, &fbh);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        if (want_exit && cli.screenshot)
            take_screenshot(*cli.screenshot);
        glfwSwapBuffers(window);
        if (want_exit)
            break;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    if (cli.bench)
        timings.print(CliArgs::bench_warmup_frames);
    return 0;
}
