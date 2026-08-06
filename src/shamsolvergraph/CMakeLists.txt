# ~~~
# SHAMROCK code for hydrodynamics
# Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
# SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
# Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
# ~~~

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamsolvergraph CXX C)

# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE shamsolvergraph_srcs CONFIGURE_DEPENDS "*.cpp")

if(SHAMROCK_USE_SHARED_LIB)
    message("-- adding shamsolvergraph target (Shared lib)")
    add_library(shamsolvergraph SHARED ${shamsolvergraph_srcs})
else()
    message("-- adding shamsolvergraph target (object lib)")
    add_library(shamsolvergraph OBJECT ${shamsolvergraph_srcs})
endif()

add_library(shamsolvergraph::shamsolvergraph ALIAS shamsolvergraph)

target_link_libraries(shamsolvergraph PUBLIC shambase)
target_link_libraries(shamsolvergraph PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(
    shamsolvergraph PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
                           "$<INSTALL_INTERFACE:include>"
)
