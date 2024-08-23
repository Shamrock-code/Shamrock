## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------


set(coverage_flags "-fprofile-instr-generate -fcoverage-mapping")

set(CMAKE_C_FLAGS_COVERAGE "${coverage_flags}")
set(CMAKE_CXX_FLAGS_COVERAGE "${coverage_flags}")
