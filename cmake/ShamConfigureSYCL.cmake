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

# check that the wanted sycl backend is in the list
set(KNOWN_SYCL_IMPLEMENTATIONS "IntelLLVM;ACPPDirect;ACPPCmake")
if((NOT ${SYCL_IMPLEMENTATION} IN_LIST KNOWN_SYCL_IMPLEMENTATIONS) OR (NOT (DEFINED SYCL_IMPLEMENTATION)))
  message(FATAL_ERROR
    "The Shamrock SYCL backend requires specifying a SYCL implementation with "
    "-DSYCL_IMPLEMENTATION=[IntelLLVM,ACPPDirect;ACPPCmake]")
endif()

message(STATUS "Chosen SYCL implementation : ${SYCL_IMPLEMENTATION}")


# use the correct script depending on the implementation
if(${SYCL_IMPLEMENTATION} STREQUAL "IntelLLVM")
  include(SYCLAdaptIntelLLVM)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPDirect")
  include(SYCLAdaptACppDirect)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPCmake")
  include(SYCLAdaptACppCmake)
else()
  message(FATAL_ERROR
    "You are asking for the shamrock sycl backend without a sycl compiler (USE_INTEL_LLVM,USE_ACPP,)")
endif()



set(SHAMROCK_LOOP_DEFAULT "PARRALEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR PARRALEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")

set(SYCL_IMPLEMENTATION "OTHER" CACHE STRING "Sycl compiler used")
set_property(CACHE SYCL_IMPLEMENTATION PROPERTY STRINGS DPCPP OPENSYCL OPENSYCL_CMAKE OTHER)




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
    list(APPEND CMAKE_SYSTEM_PROGRAM_PATH "${INTEL_LLVM_PATH}/bin")
    list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${INTEL_LLVM_PATH}/lib")
  else()
    message(FATAL_ERROR
      "INTEL_LLVM_PATH is not set, please set it to the root path of intel llvm sycl compiler")
  endif()

elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL")

  if(DEFINED ACPP_PATH)
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL_COMP_OPENSYCL")
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include")
    set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -isystem ${ACPP_PATH}/include/sycl")
    list(APPEND CMAKE_SYSTEM_PROGRAM_PATH "${ACPP_PATH}/bin")
    list(APPEND CMAKE_SYSTEM_LIBRARY_PATH "${ACPP_PATH}/lib")
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

