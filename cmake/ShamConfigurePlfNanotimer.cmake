## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- plf_nanotimer section ----")

###############################################################################
### plf_nanotimer
###############################################################################

_check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/plf_nanotimer 55e0fcb)

include_directories(external/plf_nanotimer)
