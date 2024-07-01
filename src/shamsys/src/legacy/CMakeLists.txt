
set(Sources
    sycl_handler.cpp
    sycl_mpi_interop.cpp
    cmdopt.cpp
)
set(SourcesTest
    
)
set(Headers
    sycl_handler.hpp
)
set(HeadersTest

)


target_sources(shamrock_lib
    PRIVATE
        ${Sources}
    PUBLIC
        ${Headers}
    )
    

if("${BUILD_TEST}")
target_sources(shamrock_test
    PRIVATE
     ${SourcesTest}
    PUBLIC
     ${HeadersTest}
    )
endif()


