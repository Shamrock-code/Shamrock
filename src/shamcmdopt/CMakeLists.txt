cmake_minimum_required(VERSION 3.9)

project(Shamcmdopt CXX C)

set(Sources
        src/cmdopt.cpp
        src/env.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamcmdopt target (Shared lib)")
add_library(shamcmdopt SHARED ${Sources})
else()
message("-- adding shamcmdopt target (object lib)")
add_library(shamcmdopt OBJECT ${Sources})
endif()

target_include_directories(shamcmdopt PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")

target_link_libraries(shamcmdopt PUBLIC shambase)
