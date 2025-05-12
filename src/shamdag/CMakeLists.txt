## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamdag CXX C)

set(Sources
    src/shamdag.cpp
    src/graph2dot.cpp
    src/graph2tex.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamdag target (Shared lib)")
add_library(shamdag SHARED ${Sources})
else()
message("-- adding shamdag target (object lib)")
add_library(shamdag OBJECT ${Sources})
endif()

add_library(shamdag::shamdag ALIAS shamdag)

target_link_libraries(shamdag PUBLIC shambase)
target_link_libraries(shamdag PUBLIC shamcomm)

target_include_directories(shamdag PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
