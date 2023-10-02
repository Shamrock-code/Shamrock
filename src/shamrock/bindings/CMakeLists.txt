
set(Sources
    pyShamrockCtx.cpp
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
