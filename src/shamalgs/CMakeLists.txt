
set(Sources
)
set(SourcesTest
    
)
set(Headers

)
set(HeadersTest

)

if("${BUILD_SIM}")
target_sources(shamalgs
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()


add_subdirectory(details)
add_subdirectory(atomic)
add_subdirectory(collective)

