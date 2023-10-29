## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------


set(asan_flags "-O1 -g -fsanitize=address -fno-omit-frame-pointer")

set(CMAKE_C_FLAGS_ASAN "${asan_flags}")
set(CMAKE_CXX_FLAGS_ASAN "${asan_flags}")
