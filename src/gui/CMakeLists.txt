# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

# Shamrock control GUI (GLFW + OpenGL 3), built with -DSHAMROCK_BUILD_GUI=ON.
# For now it only opens an empty window and is not linked against any Shamrock library.

cmake_minimum_required(VERSION 3.20)

message("-- adding shamrock_gui target")

# GLFW: use the system package when present, otherwise build it.
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(glfw)
endif()
# Prefer GLVND (CMP0072 NEW default), set explicitly as src/ wraps find_package with old policies
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

set(sources src/main.cpp)
add_executable(shamrock_gui ${sources})
target_link_libraries(shamrock_gui PRIVATE glfw OpenGL::GL)
# OpenGL is deprecated on macOS (10.14+) but still works; silence the per-call warnings until the
# app gets a Metal-capable backend.
if(APPLE)
    target_compile_definitions(shamrock_gui PRIVATE GL_SILENCE_DEPRECATION)
endif()
