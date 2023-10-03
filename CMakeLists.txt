## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

################################
# Cmake config
################################
cmake_minimum_required(VERSION 3.9)

# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )

option(EXPORT_COMPILE_COMMANDS "export the compile_commands.json" ON)

# Necessary for clangd to work properly
if(EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

################################
# Shamrock configuration
################################  
project(Shamrock VERSION 2023.0)



option(SHAMROCK_WITH_MPI "use MPI libraries" On)
option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)
option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
option(SHAMROCK_USE_PCH "use precompiled headers" Off)
option(SHAMROCK_USE_NVTX "use nvtx tooling" On)


######################
#checking compiler flag support 
######################
include(ShamConfigureCXX)

include(ShamBuildAsan)
include(ShamBuildCoverage)

set(ValidShamBuildType "Debug" "Release" "ASAN" "COVERAGE")
if(NOT CMAKE_BUILD_TYPE)
    #set default build to release
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${ValidShamBuildType})
endif()
if(NOT "${CMAKE_BUILD_TYPE}" IN_LIST ValidShamBuildType)
    message(FATAL_ERROR 
        "The required build type in unknown -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "please use a build type in the following list (case-sensitive) "
        "${ValidShamBuildType}")
endif()
message(STATUS "current build type : CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")


######################
# Compiler selection
######################




enable_language(C)
enable_language(CXX)


######################
#shamrock impl control
######################



if(SHAMROCK_USE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_PROFILING")
endif()

if(SHAMROCK_USE_NVTX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_NVTX")
endif()

######################
# Add external libraries
######################
add_subdirectory(src/shamunits)
add_subdirectory(src/shambackends)

if(SHAMROCK_WITH_MPI)
    include(ShamConfigureMPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHAM_CXX_MPI_FLAGS}")
endif()

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(external)

if("${BUILD_PYLIB}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LIB_BUILD")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_EXECUTABLE_BUILD")
endif()

include_directories(external/plf_nanotimer)

######################
# generate version.cpp
######################
#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/src/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/buildbot/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp "${CMAKE_CXX_FLAGS}"
  )


message("-- Shamrock config : ")
message("     SHAMROCK_USE_PROFILING : ${SHAMROCK_USE_PROFILING}")
message("     SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
message("     SHAMROCK_USE_PCH : ${SHAMROCK_USE_PCH}")
message("     SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message("     CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message("     CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")


########################################################################################
# Targets ##############################################################################
########################################################################################


######################
# Shamalgs
######################
if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamalgs target (Shared lib)")
add_library(shamalgs SHARED "")
else()
message("-- adding shamalgs target (no lib)")
add_library(shamalgs OBJECT "")
endif()

target_include_directories(shamalgs PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamalgs PRIVATE "<sycl/sycl.hpp>")
endif()

target_link_libraries(shamalgs PUBLIC MPI::MPI_C)
target_link_libraries(shamalgs PUBLIC fmt::fmt-header-only)

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamalgs PUBLIC nvtx3-cpp)
endif()

target_link_libraries(shamalgs PUBLIC shamunits)
target_link_libraries(shamalgs PUBLIC shambackends)







######################
# Shamrock Lib
######################

message("-- adding shamrock_lib target")
add_library(shamrock_lib OBJECT src/version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock_lib PRIVATE "<sycl/sycl.hpp>")
endif()

target_link_libraries(shamrock_lib PUBLIC MPI::MPI_C)
target_link_libraries(shamrock_lib PUBLIC fmt::fmt)

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamrock_lib PUBLIC nvtx3-cpp)
endif()

if("${BUILD_PYLIB}")
    target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
    target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()


target_link_libraries(shamrock_lib PUBLIC shamunits)
target_link_libraries(shamrock_lib PUBLIC shambackends)







message("-- adding shamrock target")


if("${BUILD_PYLIB}")
pybind11_add_module(shamrock  )
target_link_libraries(shamrock PUBLIC pybind11::module)
else()
add_executable(shamrock src/main.cpp )
target_link_libraries(shamrock PUBLIC pybind11::embed)
endif()

target_link_libraries(shamrock PUBLIC shamrock_lib)
target_link_libraries(shamrock PUBLIC shamalgs)


target_link_libraries(shamrock PUBLIC MPI::MPI_C)
target_link_libraries(shamrock PUBLIC fmt::fmt)
target_link_libraries(shamrock PUBLIC shamunits)
target_link_libraries(shamrock PUBLIC shambackends)



if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamrock PUBLIC nvtx3-cpp)
endif()



if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock REUSE_FROM shamrock_lib)
endif()




target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/src")
#target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/external")

#do lib build instead if building the pylib




if("${BUILD_TEST}")

    message("-- adding shamrock_test target")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test src/main_test.cpp  )
    target_link_libraries(shamrock_test PUBLIC MPI::MPI_C)

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)

    target_link_libraries(shamrock_test PUBLIC shamrock_lib)
    target_link_libraries(shamrock_test PUBLIC shamalgs)
    target_link_libraries(shamrock_test PUBLIC shamunits)
    target_link_libraries(shamrock_test PUBLIC shambackends)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")

    target_link_libraries(shamrock_test PUBLIC fmt::fmt)

    if(SHAMROCK_USE_NVTX)
        target_link_libraries(shamrock_test PUBLIC nvtx3-cpp)
    endif()



    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock_test REUSE_FROM shamrock_lib)
    endif()

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()

add_subdirectory(src)
