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
    "-DSYCL_IMPLEMENTATION=[IntelLLVM;ACPPDirect,ACPPCmake]")
endif()

set(SYCL_IMPLEMENTATION "${SYCL_IMPLEMENTATION}" CACHE STRING "Sycl implementation used")


message(STATUS "Chosen SYCL implementation : ${SYCL_IMPLEMENTATION}")

set(SHAM_CXX_SYCL_FLAGS "")

# use the correct script depending on the implementation
if(${SYCL_IMPLEMENTATION} STREQUAL "IntelLLVM")
  include(SYCLAdaptIntelLLVM)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPDirect")
  include(SYCLAdaptACppDirect)
elseif(${SYCL_IMPLEMENTATION} STREQUAL "ACPPCmake")
  include(SYCLAdaptACppCmake)
endif()



set(SHAMROCK_LOOP_DEFAULT "PARRALEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR PARRALEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")







######################
# Make CXX flags related to sycl
######################


message( " ---- Shamrock SYCL backend config ---- ")


message( "  SYCL_COMPILER : ${SYCL_COMPILER}")

message( "  sycl 2020 reduction : ${SYCL2020_FEATURE_REDUCTION}")
if(SYCL2020_FEATURE_REDUCTION)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_REDUCTION")
endif()

message( "  sycl 2020 isinf : ${SYCL2020_FEATURE_ISINF}")
if(SYCL2020_FEATURE_ISINF)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_ISINF")
endif()

message( "  sycl 2020 clz : ${SYCL2020_FEATURE_CLZ}")
if(SYCL2020_FEATURE_CLZ)
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSYCL2020_FEATURE_CLZ")
endif()


message( "  SHAMROCK_LOOP_DEFAULT : ${SHAMROCK_LOOP_DEFAULT}")
if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR_ROUND")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

message( "  SHAMROCK_LOOP_GSIZE : ${SHAMROCK_LOOP_GSIZE}")
set(SHAM_CXX_SYCL_FLAGS "${SHAM_CXX_SYCL_FLAGS} -DSHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")


message( " -------------------------------------- ")

message(STATUS "Shamrock configure SYCL backend - done")

