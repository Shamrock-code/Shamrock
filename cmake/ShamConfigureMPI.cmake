## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- MPI section ----")

option(SHAMROCK_WITH_MPI "use MPI libraries" On)

if(NOT SHAMROCK_WITH_MPI)
    message(FATAL_ERROR "SHAMROCK_WITH_MPI=Off is not supported yet!")
endif()

set(SHAM_CXX_MPI_FLAGS "-DOMPI_SKIP_MPICXX")
set(MPI_CXX_SKIP_MPICXX true)

find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI_CXX_SKIP_MPICXX : ${MPI_CXX_SKIP_MPICXX}")
message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")
message(STATUS "SHAM_CXX_MPI_FLAGS : ${SHAM_CXX_MPI_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SHAM_CXX_MPI_FLAGS}")
