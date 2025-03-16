## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message(STATUS "Shamrock configure MPI")

set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")

set(SHAM_CXX_MPI_FLAGS "-DOMPI_SKIP_MPICXX")

message(STATUS "Shamrock configure MPI - done")
