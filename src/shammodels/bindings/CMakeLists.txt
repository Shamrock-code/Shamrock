
set(Sources
)
set(SourcesTest
    
)
set(Headers
)
set(HeadersTest

)

target_sources(shamrock
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
        ${Sources} ${SourcesTest}
    PUBLIC
        ${Headers} ${HeadersTest}
    )
endif()

