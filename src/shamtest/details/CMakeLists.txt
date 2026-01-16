## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

if("${BUILD_TEST}")
target_sources(shamrock_test
PRIVATE
    TestAssert.cpp
    DataNode.cpp
    TestData.cpp
    TestAssertList.cpp
    TestDataList.cpp
    TestResult.cpp
    Test.cpp
    reporters/texTestReport.cpp
PUBLIC
)
endif()
