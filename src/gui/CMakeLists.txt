# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

# Shamrock control GUI (GLFW + OpenGL 3), built with -DSHAMROCK_BUILD_GUI=ON.
# GLFW and OpenGL are set up in cmake/ShamConfigureGUILibs.cmake.
# For now it only opens an empty window and is not linked against any Shamrock library.

cmake_minimum_required(VERSION 3.9...3.10)

message("-- adding shamrock_gui target")

set(sources src/main.cpp)
add_executable(shamrock_gui ${sources})

target_link_libraries(shamrock_gui PRIVATE glfw OpenGL::GL)
# OpenGL is deprecated on macOS (10.14+) but still works; silence the per-call warnings until the
# app gets a Metal-capable backend.
if(APPLE)
    target_compile_definitions(shamrock_gui PRIVATE GL_SILENCE_DEPRECATION)
endif()
