


set(Sources
MortonKernels.cpp
)
set(SourcesTest

)
set(Headers

)
set(HeadersTest

)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
     ${SourcesTest}
    PUBLIC
     ${HeadersTest}
    )
endif()








add_subdirectory(bindings)
