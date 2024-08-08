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
cmake_minimum_required(VERSION 3.10)

enable_testing()

# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )

# I don't know why in CI containers using an option to trigger 
# this has effect only at second configure call
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################
# Shamrock configuration
################################  
project(Shamrock VERSION 2024.0)



option(SHAMROCK_WITH_MPI "use MPI libraries" On)
option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)
option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
option(SHAMROCK_USE_PCH "use precompiled headers" Off)
option(SHAMROCK_USE_NVTX "use nvtx tooling" On)

if(APPLE)
if(SHAMROCK_USE_SHARED_LIB)
message(STATUS "forcing SHAMROCK_USE_SHARED_LIB=Off on Macos because of known issues")
endif()
set(SHAMROCK_USE_SHARED_LIB Off)
endif()

######################
#checking compiler flag support 
######################
include(ShamConfigureCXX)

include(ShamBuildAsan)
include(ShamBuildUBsan)
include(ShamBuildCoverage)

set(ValidShamBuildType "Debug" "Release" "ASAN" "UBSAN" "COVERAGE")
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

if("${SHAMROCK_ENABLE_BACKEND}" STREQUAL "SYCL")
    include(ShamConfigureSYCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHAM_CXX_SYCL_FLAGS}")
else()
    message(FATAL_ERROR
        "You must select a Shamrock Backend "
        "-DSHAMROCK_ENABLE_BACKEND=[SYCL]")
endif()
set(SHAMROCK_ENABLE_BACKEND "${SHAMROCK_ENABLE_BACKEND}" CACHE STRING "Shamrock backend used")



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
if(SHAMROCK_WITH_MPI)
    include(ShamConfigureMPI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHAM_CXX_MPI_FLAGS}")
endif()

find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(external)

if("${BUILD_PYLIB}")
    # We must force fPIC because for unknown reasons cmake doesn't on those shared lib
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LIB_BUILD -fPIC")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_EXECUTABLE_BUILD")
endif()

include_directories(external/plf_nanotimer)

message("-- Shamrock config : ")
message("     SHAMROCK_USE_PROFILING : ${SHAMROCK_USE_PROFILING}")
message("     SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
message("     SHAMROCK_USE_PCH : ${SHAMROCK_USE_PCH}")
message("     SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message("     CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message("     CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

########################################################################################
# Targets ##############################################################################
########################################################################################


add_subdirectory(src)


