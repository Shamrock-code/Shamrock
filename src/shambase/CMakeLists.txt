
set(Sources
    exception.cpp
    stacktrace.cpp
)

set(Headers

)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
