## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamcomm CXX C)

set(Sources
    src/worldInfo.cpp
    src/mpiErrorCheck.cpp
    src/mpiInfo.cpp
    src/collectives.cpp
    src/io.cpp
    src/logs.cpp
    src/wrapper.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamcomm target (Shared lib)")
add_library(shamcomm SHARED ${Sources})
else()
message("-- adding shamcomm target (object lib)")
add_library(shamcomm OBJECT ${Sources})
endif()

add_library(shamcomm::shamcomm ALIAS shamcomm)

target_link_libraries(shamcomm PUBLIC shambase)
target_link_libraries(shamcomm PUBLIC shamcmdopt)
target_link_libraries(shamcomm PUBLIC MPI::MPI_C)

target_include_directories(shamcomm PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
