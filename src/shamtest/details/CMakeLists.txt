

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
