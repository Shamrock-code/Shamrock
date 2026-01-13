## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

# This file gets called when we want to configure with AdaptiveCpp directly using cmake integration


if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(FATAL_ERROR
    "Acpp cmake integration requires Clang compiler, but ${CMAKE_CXX_COMPILER_ID} is used "
    "please set clang as cxx compiler "
    "-DCMAKE_CXX_COMPILER=<clang_path>")
endif()






if((NOT (DEFINED HIPSYCL_TARGETS)) AND (NOT (DEFINED ACPP_TARGETS)))

  message(FATAL_ERROR "no target are set for AdaptiveCpp: "
  "known valid targets are :
  omp (OpenMP backend)
  generic (Single pass compiler)
  spirv (generate spirV)
  cuda:sm_xx  (CUDA backend)
        where xx can be
        50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 90a
  hip-gfxXXX (HIP / ROCM backend)
        where XXX can be
        701, 801, 802, 803, 900, 906, 908, 1010, 1011, 1012, 1030, 1031

  please set -DHIPSYCL_TARGETS=<target_list>
  or -DACPP_TARGETS=<target_list>
  depending on the compiler version
  exemple : -DACPP_TARGETS=omp;cuda:sm_52
  ")

endif()






# try any of the ACPP current / legacy configs
find_package(AdaptiveCpp CONFIG)
if(NOT AdaptiveCpp_FOUND)

  find_package(hipSYCL CONFIG)
  if(NOT hipSYCL_FOUND)
    message(FATAL_ERROR
      "You asked shamrock to compiler using
        the acpp/opensycl/hipsycl cmake integration,
        but neither of the cmake packages can be found"    )
  else()
    set(HIPSYCL_CLANG "${CMAKE_CXX_COMPILER}")


  endif()


else()

  set(ACPP_CLANG "${CMAKE_CXX_COMPILER}")

endif()

set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_ACPP")

set(SYCL2020_FEATURE_REDUCTION Off)
set(SYCL2020_FEATURE_ISINF Off)
set(SYCL2020_FEATURE_CLZ ON)

set(SYCL_COMPILER "ACPP_CMAKE")

option(ACPP_FAST_MATH Off)

if(ACPP_FAST_MATH)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -ffast-math")
endif()

message(" ---- Acpp compiler cmake config ---- ")
message("  ACPP_FAST_MATH : ${ACPP_FAST_MATH}")
message("  HIPSYCL_TARGETS : ${HIPSYCL_TARGETS}")
message("  ACPP_TARGETS : ${ACPP_TARGETS}")
message(" ------------------------------------ ")
