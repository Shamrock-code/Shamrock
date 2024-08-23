
set(Sources
Patch.cpp
PatchData.cpp
PatchDataLayout.cpp
PatchDataField.cpp
SimBox.cpp
)
set(SourcesTest

)
set(Headers
    Patch.hpp
)
set(HeadersTest

)

target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
