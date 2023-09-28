## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)



set(SHAMROCK_LOOP_DEFAULT "PARRALEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR PARRALEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")

set(SYCL_COMPILER "OTHER" CACHE STRING "Sycl compiler used")
set_property(CACHE SYCL_COMPILER PROPERTY STRINGS DPCPP OPENSYCL OTHER)



check_cxx_source_compiles(
    "
    #include <sycl/sycl.hpp>
    int main(void){}
    " HAS_SYCL2020_HEADER)

if(NOT )
message(FATAL_ERROR "The sycl header test fail")
endif()

check_cxx_source_compiles("
#if (defined(__ACPP__) || defined(__OPENSYCL__))
int main() { return 0; }
#else
not oneAPI
#endif
" SYCL_COMPILER_IS_ACPP)
if (SYCL_COMPILER_IS_ACPP)
    set(SYCL_COMPILER "OPENSYCL")
endif()



if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  check_cxx_source_compiles("
                #include <sycl/sycl.hpp>
                #if (defined(SYCL_IMPLEMENTATION_ONEAPI))
                int main() { return 0; }
                #else
                not oneAPI
                #endif" SYCL_COMPILER_IS_DPCPP)
  if (SYCL_COMPILER_IS_DPCPP)
    set(SYCL_COMPILER "DPCPP")
  endif()
endif()







if(NOT DEFINED SYCL_feature_reduc2020)
message(STATUS "Performing Test " SYCL_feature_reduc2020)
try_compile(
    SYCL_feature_reduc2020 ${CMAKE_BINARY_DIR}/compile_tests 
    ${CMAKE_SOURCE_DIR}/cmake/feature_test/sycl2020_reduc.cpp)
endif()
if(SYCL_feature_reduc2020)
    message(STATUS "Performing Test " SYCL_feature_reduc2020 " - Success")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL2020_FEATURE_REDUCTION")
else()
message(STATUS "Performing Test " SYCL_feature_reduc2020 " - Failed")
endif()





if(${SYCL_COMPILER} STREQUAL "DPCPP")

check_cxx_compiler_flag("-fsycl-id-queries-fit-in-int" INTEL_LLVM_HAS_FIT_ID_INT)
if(INTEL_LLVM_HAS_FIT_ID_INT)
option(INTEL_LLVM_SYCL_ID_INT32 Off)
endif()

check_cxx_compiler_flag("-fno-sycl-rdc" INTEL_LLVM_HAS_NO_RDC)
if(INTEL_LLVM_HAS_NO_RDC)
option(INTEL_LLVM_NO_RDC Off)
endif()

elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL")

endif()