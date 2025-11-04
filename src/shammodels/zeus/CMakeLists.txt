## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shammodels_zeus CXX C)

set(Sources
    src/Model.cpp
    src/Solver.cpp
    src/pyAMRZeusModel.cpp

    src/modules/GhostZones.cpp
    src/modules/AMRTree.cpp
    src/modules/ComputePressure.cpp
    src/modules/FaceFlagger.cpp
    src/modules/SourceStep.cpp
    src/modules/DiffOperator.cpp
    src/modules/ValueLoader.cpp
    src/modules/WriteBack.cpp
    src/modules/TransportStep.cpp
    src/modules/AnalysisSodTube.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_zeus target (Shared lib)")
add_library(shammodels_zeus SHARED ${Sources})
else()
message("-- adding shammodels_zeus target (object lib)")
add_library(shammodels_zeus OBJECT ${Sources})
endif()

add_library(shammodels_zeus::shammodels_zeus ALIAS shammodels_zeus)

target_link_libraries(shammodels_zeus PUBLIC shambackends)
target_link_libraries(shammodels_zeus PUBLIC shamlib)
target_link_libraries(shammodels_zeus PUBLIC shammath)
target_link_libraries(shammodels_zeus PUBLIC shamphys)
target_link_libraries(shammodels_zeus PUBLIC shamsys)
target_link_libraries(shammodels_zeus PUBLIC shammodels_common)
target_link_libraries(shammodels_zeus PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_zeus PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
