## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shambase CXX C)

set(Sources
    src/exception.cpp
    src/stacktrace.cpp
    src/SourceLocation.cpp
    src/print.cpp
    src/popen.cpp
    src/profiling/chrome.cpp
    src/profiling/profiling.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shambase target (Shared lib)")
add_library(shambase SHARED ${Sources})
else()
message("-- adding shambase target (object lib)")
add_library(shambase OBJECT ${Sources})
endif()

add_library(shambase::shambase ALIAS shambase)

target_include_directories(shambase PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")



if(SHAMROCK_USE_NVTX)
    target_link_libraries(shambase PUBLIC nvtx3-cpp)
endif()

target_link_libraries(shambase PUBLIC fmt::fmt-header-only)
