


if("${BUILD_SIM}")
target_sources(shamalgs
    PRIVATE
    ResizableBuffer.cpp
    ResizableUSMBuffer.cpp
    PUBLIC
        
    )
endif()
