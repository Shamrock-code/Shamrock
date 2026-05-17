# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamterm CXX C)

set(sources src/tty.cpp src/color.cpp src/color_env.cpp)

if(SHAMROCK_USE_SHARED_LIB)
    message("-- adding shamterm target (Shared lib)")
    add_library(shamterm SHARED ${sources})
else()
    message("-- adding shamterm target (object lib)")
    add_library(shamterm OBJECT ${sources})
endif()

add_library(shamterm::shamterm ALIAS shamterm)

target_include_directories(
    shamterm PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                    "$<INSTALL_INTERFACE:include>"
)
