## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------


message(STATUS "Shamrock configure MPI")

set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")

set(SHAM_CXX_MPI_FLAGS "-DOMPI_SKIP_MPICXX")

message(STATUS "Shamrock configure MPI - done")
