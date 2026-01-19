## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

######################
# precompiled headers
######################

option(SHAMROCK_USE_PCH "use precompiled headers" Off)

######################
# Shared/Object libs
######################

option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)

if(DEFINED SHAMROCK_FORCE_SHARED_LIB)
    set(SHAMROCK_USE_SHARED_LIB ${SHAMROCK_FORCE_SHARED_LIB})
    message(WARNING "SHAMROCK_USE_SHARED_LIB was forced to ${SHAMROCK_USE_SHARED_LIB}")
endif()

# Force -fPIC for object lib mode as the python lib require it
if(NOT SHAMROCK_USE_SHARED_LIB)
    message(WARNING "using shamrock in object lib mode force the use of -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

######################
# profiling control
######################

option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
if(SHAMROCK_USE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_PROFILING")
endif()

######################
# Summary
######################

message("   ---- SUMARRY ----")

message(STATUS "CMAKE_C_COMPILER        : ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER      : ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS         : ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS  : ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "SHAMROCK_USE_PROFILING  : ${SHAMROCK_USE_PROFILING}")
message(STATUS "SHAMROCK_USE_NVTX       : ${SHAMROCK_USE_NVTX}")
message(STATUS "SHAMROCK_USE_PCH        : ${SHAMROCK_USE_PCH}")
message(STATUS "SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message(STATUS "CMAKE_BUILD_TYPE        : ${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_TEST              : ${BUILD_TEST}")
message(STATUS "PYTHON_EXECUTABLE       : ${PYTHON_EXECUTABLE}")
