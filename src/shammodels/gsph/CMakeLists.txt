## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9)

project(Shammodels_gsph CXX C)

# Sources: GSPH solver, Model, VTK I/O, and Python bindings
set(Sources
    src/SolverConfig.cpp
    src/Solver.cpp
    src/Model.cpp
    src/pyGSPHModel.cpp
    src/modules/UpdateDerivs.cpp
    src/modules/io/VTKDump.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_gsph target (Shared lib)")
add_library(shammodels_gsph SHARED ${Sources})
else()
message("-- adding shammodels_gsph target (object lib)")
add_library(shammodels_gsph OBJECT ${Sources})
endif()

add_library(shammodels_gsph::shammodels_gsph ALIAS shammodels_gsph)

target_link_libraries(shammodels_gsph PUBLIC shambackends)
target_link_libraries(shammodels_gsph PUBLIC shamlib)
target_link_libraries(shammodels_gsph PUBLIC shammath)
target_link_libraries(shammodels_gsph PUBLIC shamphys)
target_link_libraries(shammodels_gsph PUBLIC shamsys)
target_link_libraries(shammodels_gsph PUBLIC shamtree)
target_link_libraries(shammodels_gsph PUBLIC shammodels_common)
target_link_libraries(shammodels_gsph PUBLIC shammodels_sph)
target_link_libraries(shammodels_gsph PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_gsph PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
