## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

################################
# Cmake config
################################
cmake_minimum_required(VERSION 3.9...3.10)

enable_testing()

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

include(AreSubmoduleClones)

################################
# Shamrock versioning
################################

set(SHAMROCK_VERSION_MAJOR 2025)
set(SHAMROCK_VERSION_MINOR 05)
set(SHAMROCK_VERSION_PATCH 0)

project(Shamrock VERSION ${SHAMROCK_VERSION_MAJOR}.${SHAMROCK_VERSION_MINOR}.${SHAMROCK_VERSION_PATCH})
include(ShamVersioning)

######################
# Platform specfics
######################

if(APPLE)
    message(WARNING "forcing SHAMROCK_USE_SHARED_LIB=Off on Macos because of known issues")
    set(SHAMROCK_FORCE_SHARED_LIB Off)
endif()

######################
# CXX specfics
######################

include(ShamConfigureCXX)

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
# add libs
######################

include(ShamConfigureMPI)
include(ShamConfigurePybind11)
include(ShamConfigureNVTX)
include(ShamConfigureFmtlib)
include(ShamConfigureNlohmannJson)
include(ShamConfigurePlfNanotimer)
include(ShamConfigureMDSpan)

####################################################################################################
# Shamrock Configuration
####################################################################################################

include(ShamrockBuildOptions)

####################################################################################################
# target Configuration
####################################################################################################

message("   ---- targets section ----")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
