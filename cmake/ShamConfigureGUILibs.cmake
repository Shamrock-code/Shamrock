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
    ###############################################################################
    ### GLFW
    ###############################################################################

    # use the system package when present, otherwise build it.
    find_package(glfw3 3.3 QUIET)
    if(glfw3_FOUND)
        message(STATUS "GLFW : system (version ${glfw3_VERSION}, ${glfw3_DIR})")
    else()
        message(STATUS "GLFW : FetchContent (tag 3.4)")
        if(CMAKE_VERSION VERSION_LESS 3.14)
            message(
                FATAL_ERROR
                    "fetching GLFW requires CMake >= 3.14 (FetchContent_MakeAvailable), "
                    "found ${CMAKE_VERSION}. Install GLFW >= 3.3 (e.g. libglfw3-dev) or update CMake."
            )
        endif()
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

    ###############################################################################
    ### OpenGL
    ###############################################################################

    # Prefer GLVND (CMP0072 NEW default), set explicitly as the root project uses CMake 3.10 policies
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)
endif()
