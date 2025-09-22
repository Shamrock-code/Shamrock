## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- MDSPAN section ----")

###############################################################################
### MDSPAN
###############################################################################

_check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/mdspan 414a5dccf5c775b2eecc13d408b256e94f23d1d2)

include_directories(external/mdspan/include)
