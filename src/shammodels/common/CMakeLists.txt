## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shammodels_common CXX C)

set(Sources
    src/EOSConfig.cpp
    src/pyAMRTestModel.cpp
    src/pyAMRGrid.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_common target (Shared lib)")
add_library(shammodels_common SHARED ${Sources})
else()
message("-- adding shammodels_common target (object lib)")
add_library(shammodels_common OBJECT ${Sources})
endif()

add_library(shammodels_common::shammodels_common ALIAS shammodels_common)

target_link_libraries(shammodels_common PUBLIC shambackends)
target_link_libraries(shammodels_common PUBLIC shamlib)
target_link_libraries(shammodels_common PUBLIC shammath)
target_link_libraries(shammodels_common PUBLIC shamphys)
target_link_libraries(shammodels_common PUBLIC shamsys)
target_link_libraries(shammodels_common PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_common PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
