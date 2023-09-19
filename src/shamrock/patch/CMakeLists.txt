
set(Sources
Patch.cpp
PatchData.cpp
PatchDataLayout.cpp
PatchDataField.cpp
)
set(SourcesTest
    
)
set(Headers
    Patch.hpp
)
set(HeadersTest

)

if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
        ${SourcesTest}
    PUBLIC
        ${HeadersTest}
    )
endif()
