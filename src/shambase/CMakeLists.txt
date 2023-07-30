
set(Sources
    exception.cpp
    stacktrace.cpp
)

set(Headers

)

if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()
