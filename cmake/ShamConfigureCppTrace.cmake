## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- CPPTRACE section ----")

# TO enable this one do the following :
#     cmake . -DSHAMROCK_USE_CPPTRACE=on
# it may fail the autodetect of the libs to unwind, in this case you can try to set the following :
#     cmake . -DSHAMROCK_USE_CPPTRACE=on -DCPPTRACE_UNWIND_WITH_LIBUNWIND=1 -DCPPTRACE_UNWIND_WITH_LIBUNWIND=1
# see https://github.com/jeremy-rifkin/cpptrace for more configuration options

option(SHAMROCK_USE_CPPTRACE "use cpptrace tooling" Off)
message(STATUS "SHAMROCK_USE_CPPTRACE : ${SHAMROCK_USE_CPPTRACE}")

if(SHAMROCK_USE_CPPTRACE)
    ###############################################################################
    ### CPPTRACE
    ###############################################################################

    _check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/cpptrace 3db8da80111171c219ab5839905771386bee06b3)

    add_subdirectory(external/cpptrace)
endif()
