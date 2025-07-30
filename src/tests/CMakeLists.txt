## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

if("${BUILD_TEST}")

    # Whenever this glob's value changes, cmake will rerun and update the build with the
    # new/removed files.
    file(GLOB_RECURSE test_list CONFIGURE_DEPENDS "*.cpp")

    target_sources(shamrock_test PRIVATE ${test_list})

endif()
