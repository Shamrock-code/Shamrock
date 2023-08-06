

if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        pySPHModel.cpp
    PUBLIC
        
    )
endif()
