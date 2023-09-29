## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

# This file gets called when we want to configure with AdaptiveCpp directly using cmake integration

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

set(SYCL2020_FEATURE_REDUCTION Off)



set(SYCL_COMPILER "OPENSYCL_CMAKE")

