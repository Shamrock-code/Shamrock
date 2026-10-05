# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

# Shamrock control GUI (Dear ImGui docking + GLFW + OpenGL 3), built with -DSHAMROCK_BUILD_GUI=ON.
# Its dependencies are set up in cmake/ShamConfigureGUILibs.cmake.
# For now it only shows an empty dock area and is not linked against any Shamrock library.

cmake_minimum_required(VERSION 3.9...3.10)

message("-- adding shamrock_gui target")

add_library(
    shamrock_gui_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(
    shamrock_gui_imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(shamrock_gui_imgui PUBLIC glfw OpenGL::GL)
# OpenGL is deprecated on macOS (10.14+) but still works; silence the per-call warnings until the
# app gets a Metal-capable backend. PUBLIC so it also reaches shamrock_gui.
if(APPLE)
    target_compile_definitions(shamrock_gui_imgui PUBLIC GL_SILENCE_DEPRECATION)
endif()

set(sources src/main.cpp src/screenshot.cpp src/FrameTimings.cpp)
add_executable(shamrock_gui ${sources})

target_include_directories(
    shamrock_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${stb_SOURCE_DIR}
)
target_link_libraries(shamrock_gui PRIVATE shamrock_gui_imgui)
