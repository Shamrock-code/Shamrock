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

    ###############################################################################
    ### FreeType (optional)
    ###############################################################################

    # FreeType rasterises glyphs the same way as the imgui-bundle wheel.
    # Without it Dear ImGui falls back to its built-in stb_truetype.
    option(SHAMROCK_GUI_FREETYPE "Use FreeType for font rasterisation in shamrock_gui" ON)
    if(SHAMROCK_GUI_FREETYPE)
        find_package(Freetype)
    endif()

    ###############################################################################
    ### IBM Plex fonts
    ###############################################################################

    # SIL OFL 1.1, from the upstream release archives.
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(
        ibm_plex_sans
        URL https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-sans%401.1.0/ibm-plex-sans.zip
        URL_HASH SHA256=fb365d910566e6d199cc2c15579a7dd9a267128e18431a394ed81f1970c69200
    )
    FetchContent_Declare(
        ibm_plex_mono
        URL https://github.com/IBM/plex/releases/download/%40ibm%2Fplex-mono%402.5.0/ibm-plex-mono.zip
        URL_HASH SHA256=6d23f01257663d8cc49a0d64c22ced630b79e0e2a0ac08a0da86e9a38bbc481c
    )
    FetchContent_MakeAvailable(ibm_plex_sans ibm_plex_mono)
    set(SHAMROCK_GUI_FONTS
        ${ibm_plex_sans_SOURCE_DIR}/fonts/complete/ttf/IBMPlexSans-Regular.ttf
        ${ibm_plex_sans_SOURCE_DIR}/fonts/complete/ttf/IBMPlexSans-Medium.ttf
        ${ibm_plex_sans_SOURCE_DIR}/fonts/complete/ttf/IBMPlexSans-SemiBold.ttf
        ${ibm_plex_mono_SOURCE_DIR}/fonts/complete/ttf/IBMPlexMono-Regular.ttf
        ${ibm_plex_mono_SOURCE_DIR}/fonts/complete/ttf/IBMPlexMono-Medium.ttf
    )
    set(SHAMROCK_GUI_FONTS_LICENSE ${ibm_plex_sans_SOURCE_DIR}/LICENSE.txt)
endif()
