## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- NVTX section ----")

option(SHAMROCK_USE_NVTX "use nvtx tooling" On)

###############################################################################
### NVTX
###############################################################################

if(SHAMROCK_USE_NVTX)
    #include(NVTX/c/nvtxImportedTargets.cmake)

    _check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/NVTX b44f81c)

    add_subdirectory(external/NVTX/c)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_NVTX")
endif()

message(STATUS "SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
