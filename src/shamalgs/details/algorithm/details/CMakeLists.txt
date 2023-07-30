
set(Sources
bitonicSort_legacy.cpp
bitonicSort_updated.cpp
#bitonicSort_updated_xor_swap.cpp
)
set(SourcesTest
    
)
set(Headers

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
