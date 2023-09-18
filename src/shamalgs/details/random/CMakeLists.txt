
set(Sources
random.cpp
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

