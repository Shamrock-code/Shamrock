## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(ubsan_flags "-O1 -g -fsanitize=undefined -fno-omit-frame-pointer")

set(CMAKE_C_FLAGS_UBSAN "${ubsan_flags}")
set(CMAKE_CXX_FLAGS_UBSAN "${ubsan_flags}")
