## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shammath CXX C)

set(Sources
    src/CoordRangeTransform.cpp
    src/CoordRange.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammath target (Shared lib)")
add_library(shammath SHARED ${Sources})
else()
message("-- adding shammath target (object lib)")
add_library(shammath OBJECT ${Sources})
endif()

add_library(shammath::shammath ALIAS shammath)

target_link_libraries(shammath PUBLIC shambackends)

target_include_directories(shammath PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
