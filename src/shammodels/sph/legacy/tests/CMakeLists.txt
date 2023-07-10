
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    benchmark_soundwave.cpp
    PUBLIC
         
    )


endif()
