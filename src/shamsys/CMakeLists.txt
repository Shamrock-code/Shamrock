cmake_minimum_required(VERSION 3.9)

project(Shamsys CXX C)

set(Sources
        src/NodeInstance.cpp
        src/SyclHelper.cpp
        src/SignalCatch.cpp
        src/MicroBenchmark.cpp
        src/legacy/sycl_handler.cpp
        src/legacy/sycl_mpi_interop.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamsys target (Shared lib)")
add_library(shamsys SHARED ${Sources})
else()
message("-- adding shamsys target (object lib)")
add_library(shamsys OBJECT ${Sources})
endif()

add_library(shamsys::shamsys ALIAS shamsys)

target_link_libraries(shamsys PUBLIC shambase)
target_link_libraries(shamsys PUBLIC shamcomm)
target_link_libraries(shamsys PUBLIC shambackends)
target_link_libraries(shamsys PUBLIC shamalgs) #TODO shamsys should not depends on algs
target_link_libraries(shamsys PUBLIC shamcmdopt)

target_include_directories(shamsys PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
