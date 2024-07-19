## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(ubsan_flags "-O1 -g -fsanitize=undefined -fno-omit-frame-pointer")

set(CMAKE_C_FLAGS_UBSAN "${ubsan_flags}")
set(CMAKE_CXX_FLAGS_UBSAN "${ubsan_flags}")
