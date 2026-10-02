# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

message("   ---- GUI libs section ----")

# Standalone control GUI (not linked against Shamrock yet), enable it with :
#     cmake . -DSHAMROCK_BUILD_GUI=on

option(SHAMROCK_BUILD_GUI "build the standalone shamrock_gui executable" Off)
message(STATUS "SHAMROCK_BUILD_GUI : ${SHAMROCK_BUILD_GUI}")

if(SHAMROCK_BUILD_GUI)
    # the FetchContent_Declare / FetchContent_MakeAvailable calls below require CMake >= 3.14.
    if(CMAKE_VERSION VERSION_LESS 3.14)
        message(
            FATAL_ERROR
                "the GUI fetches its dependencies with FetchContent_MakeAvailable, which requires "
                "CMake >= 3.14, found ${CMAKE_VERSION}. Update CMake or build without the GUI."
        )
    endif()
    include(FetchContent)

    ###############################################################################
    ### GLFW
    ###############################################################################

    # use the system package when present, otherwise build it.
    find_package(glfw3 3.3 QUIET)
    if(glfw3_FOUND)
        message(STATUS "GLFW : system (version ${glfw3_VERSION}, ${glfw3_DIR})")
    else()
        message(STATUS "GLFW : FetchContent (tag 3.4)")
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

    ###############################################################################
    ### OpenGL
    ###############################################################################

    # Prefer GLVND (CMP0072 NEW default), set explicitly as the root project uses CMake 3.10 policies
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)

    ###############################################################################
    ### Dear ImGui (docking)
    ###############################################################################

    # Docking build of the Dear ImGui version used by imgui-bundle 1.92.900.
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.92.9-docking
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui)

    ###############################################################################
    ### stb
    ###############################################################################

    # stb_image_write, used to save --screenshot PNGs.
    FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(stb)
endif()
