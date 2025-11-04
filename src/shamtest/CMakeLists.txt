## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(SourcesTest
    shamtest.cpp
)
set(HeadersTest
    shamtest.hpp
)

if("${BUILD_TEST}")
target_sources(shamrock_test
PRIVATE
    ${SourcesTest}
PUBLIC
    ${HeadersTest}
)
endif()

add_subdirectory(details)
