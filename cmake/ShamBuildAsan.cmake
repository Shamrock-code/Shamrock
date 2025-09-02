## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(asan_flags "-O1 -g -fsanitize=address -fno-omit-frame-pointer")

set(CMAKE_C_FLAGS_ASAN "${asan_flags}")
set(CMAKE_CXX_FLAGS_ASAN "${asan_flags}")

if(CMAKE_BUILD_TYPE STREQUAL "ASAN")
    if(NOT DEFINED SHAM_ASSERT_MODE_DEFAULT)
        set(SHAM_ASSERT_MODE_DEFAULT RUNTIME_ERROR)
        message(STATUS "Setting SHAM_ASSERT_MODE_DEFAULT=RUNTIME_ERROR with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} (you can force it off).")
    endif()
endif()
