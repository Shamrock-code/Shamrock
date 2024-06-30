cmake_minimum_required(VERSION 3.9)

project(Shamalgs CXX C)

set(Sources
        src/serialize.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamalgs target (Shared lib)")
add_library(shamalgs SHARED ${Sources})
else()
message("-- adding shamalgs target (object lib)")
add_library(shamalgs OBJECT ${Sources})
endif()

add_library(shamalgs::shamalgs ALIAS shamalgs)

target_link_libraries(shamalgs PUBLIC shambase)
target_link_libraries(shamalgs PUBLIC shamcomm)
target_link_libraries(shamalgs PUBLIC shambackends)

target_include_directories(shamalgs PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
