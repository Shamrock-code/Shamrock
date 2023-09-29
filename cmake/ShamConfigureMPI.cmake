

message(STATUS "Shamrock configure MPI")

set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")

set(SHAM_CXX_MPI_FLAGS "-DOMPI_SKIP_MPICXX")

message(STATUS "Shamrock configure MPI - done")
