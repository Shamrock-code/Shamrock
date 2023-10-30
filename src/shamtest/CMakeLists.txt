
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
