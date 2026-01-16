## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamcmdopt CXX C)

set(Sources
        src/details/generic_opts.cpp
        src/cmdopt.cpp
        src/env.cpp
        src/ci_env.cpp
        src/tty.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamcmdopt target (Shared lib)")
add_library(shamcmdopt SHARED ${Sources})
else()
message("-- adding shamcmdopt target (object lib)")
add_library(shamcmdopt OBJECT ${Sources})
endif()

target_link_libraries(shamcmdopt PUBLIC shambase)

target_include_directories(shamcmdopt PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
