## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(coverage_flags "-fprofile-instr-generate -fcoverage-mapping")

set(CMAKE_C_FLAGS_COVERAGE "${coverage_flags}")
set(CMAKE_CXX_FLAGS_COVERAGE "${coverage_flags}")

# disable shared lib if coverage is enabled
if(CMAKE_BUILD_TYPE STREQUAL "COVERAGE")
    if(SHAMROCK_USE_SHARED_LIB)
    message(STATUS
        "forcing SHAMROCK_USE_SHARED_LIB=Off with coverage enabled, "
        "without it llvm-cov will show only the main binary functions")
    endif()
    set(SHAMROCK_FORCE_SHARED_LIB Off)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "COVERAGE")
    if(NOT DEFINED SHAM_ASSERT_MODE_DEFAULT)
        set(SHAM_ASSERT_MODE_DEFAULT RUNTIME_ERROR)
        message(STATUS "Setting SHAM_ASSERT_MODE_DEFAULT=RUNTIME_ERROR with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} (you can force it off).")
    endif()
endif()
