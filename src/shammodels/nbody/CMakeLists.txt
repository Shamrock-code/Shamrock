## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shammodels_nbody CXX C)

set(Sources
    src/models/nbody_selfgrav.cpp
    src/setup/nbody_setup.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_nbody target (Shared lib)")
add_library(shammodels_nbody SHARED ${Sources})
else()
message("-- adding shammodels_nbody target (object lib)")
add_library(shammodels_nbody OBJECT ${Sources})
endif()

add_library(shammodels_nbody::shammodels_nbody ALIAS shammodels_nbody)

target_link_libraries(shammodels_nbody PUBLIC shambackends)
target_link_libraries(shammodels_nbody PUBLIC shamlib)
target_link_libraries(shammodels_nbody PUBLIC shammath)
target_link_libraries(shammodels_nbody PUBLIC shamphys)
target_link_libraries(shammodels_nbody PUBLIC shamsys)
target_link_libraries(shammodels_nbody PUBLIC shammodels_common)
target_link_libraries(shammodels_nbody PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_nbody PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
