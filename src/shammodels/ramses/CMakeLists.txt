cmake_minimum_required(VERSION 3.9)

project(Shammodels_ramses CXX C)

set(Sources
    src/Model.cpp
    src/pyAMRGodunovModel.cpp
    src/Solver.cpp

    src/modules/AMRGraphGen.cpp
    src/modules/AMRTree.cpp
    src/modules/ComputeFlux.cpp
    src/modules/DragIntegrator.cpp
    src/modules/TimeIntegrator.cpp
    src/modules/AMRGridRefinementHandler.cpp
    src/modules/AnalysisSodTube.cpp
    src/modules/ComputeGradient.cpp
    src/modules/FaceInterpolate.cpp
    src/modules/AMRSetup.cpp
    src/modules/ComputeCellInfos.cpp
    src/modules/ComputeTimeDerivative.cpp
    src/modules/GhostZones.cpp
    src/modules/AMRSortBlocks.cpp
    src/modules/ComputeCFL.cpp
    src/modules/ConsToPrim.cpp
    src/modules/StencilGenerator.cpp
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
