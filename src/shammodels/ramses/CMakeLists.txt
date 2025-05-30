## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shammodels_ramses CXX C)

set(Sources
    src/Model.cpp
    src/pyRamsesModel.cpp
    src/Solver.cpp

    src/modules/ComputeFlux.cpp
    src/modules/DragIntegrator.cpp
    src/modules/TimeIntegrator.cpp
    src/modules/AMRGridRefinementHandler.cpp
    src/modules/AnalysisSodTube.cpp
    src/modules/FaceInterpolate.cpp
    src/modules/AMRSetup.cpp
    src/modules/ComputeTimeDerivative.cpp
    src/modules/GhostZones.cpp
    src/modules/AMRSortBlocks.cpp
    src/modules/ComputeCFL.cpp
    src/modules/ConsToPrimGas.cpp
    src/modules/ConsToPrimDust.cpp
    src/modules/ComputeCellAABB.cpp
    src/modules/StencilGenerator.cpp
    src/modules/NodeBuildTrees.cpp
    src/modules/FindBlockNeigh.cpp
    src/modules/BlockNeighToCellNeigh.cpp
    src/modules/SlopeLimitedGradient.cpp
    src/modules/ComputeRhoMean.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_ramses target (Shared lib)")
add_library(shammodels_ramses SHARED ${Sources})
else()
message("-- adding shammodels_ramses target (object lib)")
add_library(shammodels_ramses OBJECT ${Sources})
endif()

add_library(shammodels_ramses::shammodels_ramses ALIAS shammodels_ramses)

target_link_libraries(shammodels_ramses PUBLIC shambackends)
target_link_libraries(shammodels_ramses PUBLIC shamlib)
target_link_libraries(shammodels_ramses PUBLIC shammath)
target_link_libraries(shammodels_ramses PUBLIC shamphys)
target_link_libraries(shammodels_ramses PUBLIC shamsys)
target_link_libraries(shammodels_ramses PUBLIC shammodels_common)
target_link_libraries(shammodels_ramses PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_ramses PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
