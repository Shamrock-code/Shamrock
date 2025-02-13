cmake_minimum_required(VERSION 3.9)

project(Shammodels_sph CXX C)

set(Sources
    src/BasicSPHGhosts.cpp
    src/SPHSolverImpl.cpp
    src/Model.cpp
    src/Solver.cpp
    src/SPHUtilities.cpp
    src/SolverLog.cpp
    src/io/PhantomDump.cpp
    src/SolverConfig.cpp
    src/pySPHModel.cpp
    src/pySPHKernels.cpp
    src/pyPhantomDump.cpp

    src/modules/SolverStorage.cpp
    src/modules/BuildTrees.cpp
    src/modules/UpdateViscosity.cpp
    src/modules/DiffOperator.cpp
    src/modules/DiffOperatorDtDivv.cpp
    src/modules/SinkParticlesUpdate.cpp
    src/modules/ConservativeCheck.cpp
    src/modules/ExternalForces.cpp
    src/modules/ComputeEos.cpp
    src/modules/UpdateDerivs.cpp
    src/modules/ComputeLoadBalanceValue.cpp
    src/modules/ComputeOmega.cpp
    src/modules/NeighbourCache.cpp
    src/modules/ParticleReordering.cpp
    src/modules/AnalysisSodTube.cpp
    src/modules/SPHSetup.cpp

    src/modules/setup/GeneratorMCDisc.cpp
    src/modules/setup/ModifierApplyDiscWarp.cpp
    src/modules/render/CartesianRender.cpp
    src/modules/render/RenderFieldGetter.cpp

    src/modules/io/VTKDump.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shammodels_sph target (Shared lib)")
add_library(shammodels_sph SHARED ${Sources})
else()
message("-- adding shammodels_sph target (object lib)")
add_library(shammodels_sph OBJECT ${Sources})
endif()

add_library(shammodels_sph::shammodels_sph ALIAS shammodels_sph)

target_link_libraries(shammodels_sph PUBLIC shambackends)
target_link_libraries(shammodels_sph PUBLIC shamlib)
target_link_libraries(shammodels_sph PUBLIC shammath)
target_link_libraries(shammodels_sph PUBLIC shamphys)
target_link_libraries(shammodels_sph PUBLIC shamsys)
target_link_libraries(shammodels_sph PUBLIC shammodels_common)
target_link_libraries(shammodels_sph PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shammodels_sph PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
