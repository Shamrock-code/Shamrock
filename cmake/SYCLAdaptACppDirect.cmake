## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

# This file gets called when we want to configure against the AdaptiveCpp directly (not with cmake integration)

check_cxx_source_compiles("
    #if (defined(__ACPP__) || defined(__OPENSYCL__) || defined(__HIPSYCL__))
    int main() { return 0; }
    #else
    #error
    #endif
    "    
    SYCL_COMPILER_IS_ACPP)

if(NOT SYCL_COMPILER_IS_ACPP)
  message(FATAL_ERROR
    "ACpp does not define any of the following Macro here "
    "__ACPP__,__OPENSYCL__,__HIPSYCL__  "
    "this doesn't seems like the acpp compiler "
    "please select the acpp compiler using : "
    "-DCMAKE_CXX_COMPILER=<path_to_compiler>")
endif()

check_cxx_source_compiles(
    "
    #include <sycl/sycl.hpp>
    int main(void){}
    "    
    HAS_SYCL2020_HEADER)

if(NOT HAS_SYCL2020_HEADER)
  message(FATAL_ERROR "Acpp can not compile a simple exemple including <sycl/sycl.hpp>")
endif()

if(NOT DEFINED SYCL_feature_reduc2020)
  message(STATUS "Performing Test " SYCL_feature_reduc2020)
  try_compile(
    SYCL_feature_reduc2020 ${CMAKE_BINARY_DIR}/compile_tests
    ${CMAKE_SOURCE_DIR}/cmake/feature_test/sycl2020_reduc.cpp)
  if(SYCL_feature_reduc2020)
    message(STATUS "Performing Test " SYCL_feature_reduc2020 " - Success")
    set(SYCL2020_FEATURE_REDUCTION ON)
  else()
    message(STATUS "Performing Test " SYCL_feature_reduc2020 " - Failed")
    set(SYCL2020_FEATURE_REDUCTION Off)
  endif()
endif()


set(SYCL_COMPILER "OPENSYCL")

if(DEFINED ACPP_PATH)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_OPENSYCL")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include/sycl")
  list(APPEND CMAKE_SYSTEM_PROGRAM_PATH "${ACPP_PATH}/bin")
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${ACPP_PATH}/lib")
else()
  message(FATAL_ERROR
    "ACPP_PATH is not set, please set it to the root path of acpp (formely Hipsycl or Opensycl) sycl compiler please set "
    "-DACPP_PATH=<path_to_compiler_root_dir>")
endif()



check_cxx_compiler_flag("-ffast-math" ACPP_HAS_FAST_MATH)
if(ACPP_HAS_FAST_MATH)
    option(ACPP_FAST_MATH Off)
endif()

if(ACPP_FAST_MATH)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -ffast-math")
endif()



message(" ---- Acpp compiler direct config ---- ")
message("  ACPP_FAST_MATH : ${ACPP_FAST_MATH}")
message(" ------------------------------------- ")

