
set(Sources
    NodeInstance.cpp
    SyclMpiTypes.cpp
    SyclHelper.cpp
    SignalCatch.cpp
    EnvVariables.cpp
    MicroBenchmark.cpp
)
set(SourcesTest
    
)
set(Headers
    NodeInstance.hpp
    SyclMpiTypes.hpp
    SyclHelper.hpp
)
set(HeadersTest

)

if("${BUILD_SIM}")
target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
endif()

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
        ${SourcesTest}
    PUBLIC
        ${HeadersTest}
    )
endif()

add_subdirectory(tests)
add_subdirectory(legacy)

add_subdirectory(bindings)
add_subdirectory(comm)
