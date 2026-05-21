# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamformat CXX C)

set(sources src/format_exception.cpp)

if(SHAMROCK_USE_SHARED_LIB)
    message("-- adding shamformat target (Shared lib)")
    add_library(shamformat SHARED ${sources})
else()
    message("-- adding shamformat target (object lib)")
    add_library(shamformat OBJECT ${sources})
endif()

add_library(shamformat::shamformat ALIAS shamformat)

target_include_directories(
    shamformat PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                      "$<INSTALL_INTERFACE:include>"
)

# Lib linking

target_link_libraries(shamformat PUBLIC fmt::fmt-header-only)
