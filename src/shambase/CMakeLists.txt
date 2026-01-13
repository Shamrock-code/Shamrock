## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
    src/term_colors.cpp
    src/profiling/chrome.cpp
    src/profiling/profiling.cpp
    src/logs/reformat_message.cpp
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

# Assertion handling

if(NOT DEFINED SHAM_ASSERT_MODE_DEFAULT)
    set(SHAM_ASSERT_MODE_DEFAULT NONE)
endif()

set(ValidSHAM_ASSERT_MODE "NONE" "CASSERT" "RUNTIME_ERROR")
set(SHAM_ASSERT_MODE ${SHAM_ASSERT_MODE_DEFAULT} CACHE STRING "Shamrock mode to handle asserts")
set_property(CACHE SHAM_ASSERT_MODE PROPERTY STRINGS ${ValidSHAM_ASSERT_MODE})
if(NOT "${SHAM_ASSERT_MODE}" IN_LIST ValidSHAM_ASSERT_MODE)
    message(FATAL_ERROR
        "The shamrock assert type is unknown -DSHAM_ASSERT_MODE=${SHAM_ASSERT_MODE}. "
        "please use an assert type in the following list (case-sensitive) "
        "${ValidSHAM_ASSERT_MODE}")
endif()
message(STATUS "current assert type : SHAM_ASSERT_MODE=${SHAM_ASSERT_MODE}")

if(SHAM_ASSERT_MODE STREQUAL "NONE")
    # do nothing
elseif(SHAM_ASSERT_MODE STREQUAL "CASSERT")
target_compile_definitions(shambase PUBLIC SHAM_ASSERT_IS=ASSERT_MODE_CASSERT)
elseif(SHAM_ASSERT_MODE STREQUAL "RUNTIME_ERROR")
    target_compile_definitions(shambase PUBLIC SHAM_ASSERT_IS=ASSERT_MODE_RUNTIME_ERROR)
endif()

# Lib linking

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shambase PUBLIC nvtx3-cpp)
endif()

target_link_libraries(shambase PUBLIC fmt::fmt-header-only)
