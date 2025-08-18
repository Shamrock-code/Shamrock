## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamalgs CXX C)

set(Sources
        src/container/BufferEventHandler.cpp
        src/details/numeric/streamCompactExclScan.cpp
        src/details/numeric/exclusiveScanGPUGems39.cpp
        src/details/numeric/numeric.cpp
        src/details/numeric/numericFallback.cpp
        src/details/memory/avoidCopyMemory.cpp
        src/details/memory/memory.cpp
        src/details/memory/fallbackMemory.cpp
        src/details/algorithm/bitonicSort_updated_xor_swap.cpp
        src/details/algorithm/algorithm.cpp
        src/details/algorithm/bitonicSort_updated.cpp
        src/details/algorithm/bitonicSort_updated_usm.cpp
        src/details/algorithm/bitonicSort_legacy.cpp
        src/details/random/random.cpp
        src/details/reduction/reduction.cpp
        src/details/reduction/groupReduction_usm.cpp
        src/details/reduction/fallbackReduction_usm.cpp
        src/serialize.cpp
        src/collective/distributedDataComm.cpp
        src/collective/sparseXchg.cpp
        src/primitives/sort_by_keys.cpp
        src/primitives/is_all_true.cpp
        src/primitives/mock_value.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamalgs target (Shared lib)")
add_library(shamalgs SHARED ${Sources})
else()
message("-- adding shamalgs target (object lib)")
add_library(shamalgs OBJECT ${Sources})
endif()

add_library(shamalgs::shamalgs ALIAS shamalgs)

target_link_libraries(shamalgs PUBLIC shambackends)

target_include_directories(shamalgs PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
