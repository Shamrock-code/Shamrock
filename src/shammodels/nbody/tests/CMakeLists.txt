
if("${BUILD_TEST}")

    target_sources(shamrock_test
    PRIVATE
    benchmark_selfgrav.cpp
    PUBLIC

    )


endif()
