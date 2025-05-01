## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shamtree CXX C)

set(Sources
    src/RadixTreeMortonBuilder.cpp
    src/MortonKernels.cpp
    src/RadixTree.cpp
    src/kernels/compute_ranges.cpp
    src/kernels/key_morton_sort.cpp
    src/kernels/convert_ranges.cpp
    src/kernels/karras_alg.cpp
    src/kernels/reduction_alg.cpp
    src/MortonCodeSet.cpp
    src/MortonCodeSortedSet.cpp
    src/MortonReducedSet.cpp
    src/KarrasRadixTree.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamtree target (Shared lib)")
add_library(shamtree SHARED ${Sources})
else()
message("-- adding shamtree target (object lib)")
add_library(shamtree OBJECT ${Sources})
endif()

add_library(shamtree::shamtree ALIAS shamtree)

target_link_libraries(shamtree PUBLIC shamalgs)
target_link_libraries(shamtree PUBLIC shammath)
target_link_libraries(shamtree PUBLIC shamsys)

target_include_directories(shamtree PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
