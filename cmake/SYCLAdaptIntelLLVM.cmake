## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
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
    SYCL_COMPILER_IS_DPCPP)

if(NOT SYCL_COMPILER_IS_DPCPP)
message(FATAL_ERROR
    "intel llvm should have sycl header and defines SYCL_IMPLEMENTATION_ONEAPI, this is not the case here")
endif()

set(SYCL_COMPILER "DPCPP")

set(SYCL2020_FEATURE_REDUCTION ON)

