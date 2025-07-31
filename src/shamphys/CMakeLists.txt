## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamphys CXX C)

set(Sources
    src/HydroSoundwave.cpp
    src/SodTube.cpp
    src/SedovTaylor.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamphys target (Shared lib)")
add_library(shamphys SHARED ${Sources})
else()
message("-- adding shamphys target (object lib)")
add_library(shamphys OBJECT ${Sources})
endif()

add_library(shamphys::shamphys ALIAS shamphys)

target_link_libraries(shamphys PUBLIC shammath)
target_link_libraries(shamphys PUBLIC shamunits)
target_link_libraries(shamphys PUBLIC shambindings)

target_include_directories(shamphys PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
