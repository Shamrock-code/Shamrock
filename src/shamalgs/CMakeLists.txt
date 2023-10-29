
set(Sources
)
set(SourcesTest
    
)
set(Headers

)
set(HeadersTest

)

target_sources(shamalgs
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )


add_subdirectory(details)
add_subdirectory(atomic)
add_subdirectory(collective)
add_subdirectory(container)

