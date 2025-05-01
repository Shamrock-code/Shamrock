## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

# This file gets called when we want to configure against the intel llvm with sycl support

check_cxx_source_compiles("
      #include <sycl/sycl.hpp>
      #if (defined(SYCL_IMPLEMENTATION_ONEAPI))
      int main() { return 0; }
      #else
      #error
      #endif"
    SYCL_COMPILER_IS_INTEL_LLVM)

if(NOT SYCL_COMPILER_IS_INTEL_LLVM)
    message(FATAL_ERROR
        "intel llvm should have sycl header and defines SYCL_IMPLEMENTATION_ONEAPI, this is not the case here")
endif()

set(SYCL_COMPILER "INTEL_LLVM")

set(SYCL2020_FEATURE_REDUCTION ON)
set(SYCL2020_FEATURE_ISINF ON)
set(SYCL2020_FEATURE_CLZ ON)
set(SYCL2020_FEATURE_GROUP_REDUCTION ON)

if(DEFINED INTEL_LLVM_PATH)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_INTEL_LLVM -Wno-unknown-cuda-version")

    if(SHAMROCK_ADD_SYCL_INCLUDES)
        set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${INTEL_LLVM_PATH}/include")
        set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${INTEL_LLVM_PATH}/include/sycl")
    endif()

    list(APPEND CMAKE_SYSTEM_PROGRAM_PATH "${INTEL_LLVM_PATH}/bin")
    list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${INTEL_LLVM_PATH}/lib")
else()
    message(FATAL_ERROR
        "INTEL_LLVM_PATH is not set, please set it to the root path of intel's llvm sycl compiler please set "
        "-DINTEL_LLVM_PATH=<path_to_compiler_root_dir>")
endif()

check_cxx_compiler_flag("-fsycl-id-queries-fit-in-int" INTEL_LLVM_HAS_FIT_ID_INT)
if(INTEL_LLVM_HAS_FIT_ID_INT)
    option(INTEL_LLVM_SYCL_ID_INT32 Off)
endif()

check_cxx_compiler_flag("-fno-sycl-rdc" INTEL_LLVM_HAS_NO_RDC)
if(INTEL_LLVM_HAS_NO_RDC)
    option(INTEL_LLVM_NO_RDC Off)
endif()

check_cxx_compiler_flag("-ffast-math" INTEL_LLVM_HAS_FAST_MATH)
if(INTEL_LLVM_HAS_FAST_MATH)
    option(INTEL_LLVM_FAST_MATH Off)
endif()

if(INTEL_LLVM_FAST_MATH)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -ffast-math")
endif()

if(INTEL_LLVM_SYCL_ID_INT32)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -fsycl-id-queries-fit-in-int")
endif()

if(INTEL_LLVM_NO_RDC)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -fno-sycl-rdc")
endif()


message(" ---- Intel llvm compiler  config ---- ")
message("  INTEL_LLVM_FAST_MATH : ${INTEL_LLVM_FAST_MATH}")
message("  INTEL_LLVM_SYCL_ID_INT32 : ${INTEL_LLVM_SYCL_ID_INT32}")
message("  INTEL_LLVM_NO_RDC : ${INTEL_LLVM_NO_RDC}")
message(" ------------------------------------- ")
