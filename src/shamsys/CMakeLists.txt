## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamsys CXX C)

set(Sources
        src/NodeInstance.cpp
        src/SyclHelper.cpp
        src/SignalCatch.cpp
        src/MicroBenchmark.cpp
        src/change_log_format.cpp
        src/device_select.cpp
        src/shamrock_smi.cpp
        src/legacy/sycl_handler.cpp
        src/legacy/sycl_mpi_interop.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamsys target (Shared lib)")
add_library(shamsys SHARED ${Sources})
else()
message("-- adding shamsys target (object lib)")
add_library(shamsys OBJECT ${Sources})
endif()

add_library(shamsys::shamsys ALIAS shamsys)

target_link_libraries(shamsys PUBLIC shambackends)
target_link_libraries(shamsys PUBLIC shamalgs) #TODO shamsys should not depends on algs
target_link_libraries(shamsys PUBLIC shamcmdopt)

target_include_directories(shamsys PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
