## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)



message(STATUS "Shamrock configure SYCL backend")

set(SHAMROCK_LOOP_DEFAULT "PARRALEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR PARRALEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")

set(SYCL_COMPILER "OTHER" CACHE STRING "Sycl compiler used")
set_property(CACHE SYCL_COMPILER PROPERTY STRINGS DPCPP OPENSYCL OPENSYCL_CMAKE OTHER)





option(SYCL2020_FEATURE_REDUCTION off)



if(USE_ACPP_CMAKE)

  message(FATAL_ERROR
          "The acpp cmake integration is not (yet) supported in shamrock"        )

  # try any of the ACPP current / legacy configs
  find_package(AdaptiveCpp CONFIG)
  if(NOT AdaptiveCpp_FOUND)
    find_package(OpenSYCL CONFIG)
    if(NOT OpenSYCL_FOUND)
      find_package(hipSYCL CONFIG)
      if(NOT hipSYCL_FOUND)
        message(FATAL_ERROR
          "You asked shamrock to compiler using
          the acpp/opensycl/hipsycl cmake integration, 
          but neither of the cmake packages can be found"        )
      endif()
    endif()
  endif()

  set(SYCL_COMPILER "OPENSYCL_CMAKE")

  set(SYCL2020_FEATURE_REDUCTION Off)


#user explicitely specified that intel llvm is used
elseif(USE_INTEL_LLVM)

  check_cxx_source_compiles("
      #include <sycl/sycl.hpp>
      #if (defined(SYCL_IMPLEMENTATION_ONEAPI))
      int main() { return 0; }
      #else
      not oneAPI
      #endif"
    SYCL_COMPILER_IS_DPCPP)

  if(NOT SYCL_COMPILER_IS_DPCPP)
    message(FATAL_ERROR
      "The used compiler does not have sycl header and defines SYCL_IMPLEMENTATION_ONEAPI")
  endif()

  set(SYCL2020_FEATURE_REDUCTION ON)

#Let the config autodect the sycl compiler
else()

  check_cxx_source_compiles(
    "
    #include <sycl/sycl.hpp>
    int main(void){}
    "    
    HAS_SYCL2020_HEADER)

  if(NOT HAS_SYCL2020_HEADER)
    message(FATAL_ERROR "The selected compiler does <sycl/sycl.hpp>")
  endif()

  check_cxx_source_compiles("
    #if (defined(__ACPP__) || defined(__OPENSYCL__))
    int main() { return 0; }
    #else
    not oneAPI
    #endif
    "    
    SYCL_COMPILER_IS_ACPP)

  if(SYCL_COMPILER_IS_ACPP)
    set(SYCL_COMPILER "OPENSYCL")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    check_cxx_source_compiles("
                #include <sycl/sycl.hpp>
                #if (defined(SYCL_IMPLEMENTATION_ONEAPI))
                int main() { return 0; }
                #else
                not oneAPI
                #endif"      SYCL_COMPILER_IS_DPCPP)
    if(SYCL_COMPILER_IS_DPCPP)
      set(SYCL_COMPILER "DPCPP")
    endif()
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




######################
# Make CXX flags related to sycl
######################
set(SHAM_CXX_SYCL_FLAGS "")

message(STATUS "SYCL compiler config :")


message(STATUS "  SYCL_COMPILER : ${SYCL_COMPILER}")
if(${SYCL_COMPILER} STREQUAL "DPCPP")

  if(DEFINED INTEL_LLVM_PATH)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_DPCPP -Wno-unknown-cuda-version")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${INTEL_LLVM_PATH}/include")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${INTEL_LLVM_PATH}/include/sycl")
  list(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${INTEL_LLVM_PATH}/bin")
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${INTEL_LLVM_PATH}/lib")
  else()
  message(FATAL_ERROR 
  "INTEL_LLVM_PATH is not set, please set it to the root path of intel llvm sycl compiler")
  endif()

elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL")

  if(DEFINED ACPP_PATH)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_OPENSYCL")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include/sycl")
  list(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${ACPP_PATH}/bin")
  list(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${ACPP_PATH}/lib")
  else()
  message(FATAL_ERROR 
  "ACPP_PATH is not set, please set it to the root path of acpp (formely Hipsycl or Opensycl) sycl compiler")
  endif()

elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL_CMAKE")

  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_OPENSYCL")

elseif(${SYCL_COMPILER} STREQUAL "OTHER")

  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_SYCLUNKNOWN")

endif()


message(STATUS "  sycl 2020 reduction : ${SYCL2020_FEATURE_REDUCTION}")
if(SYCL2020_FEATURE_REDUCTION)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_REDUCTION")
endif()






message(STATUS "  SHAMROCK_LOOP_DEFAULT : ${SHAMROCK_LOOP_DEFAULT}")
if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR_ROUND")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

message(STATUS "  SHAMROCK_LOOP_GSIZE : ${SHAMROCK_LOOP_GSIZE}")
set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")



message(STATUS "Shamrock configure SYCL backend - done")

