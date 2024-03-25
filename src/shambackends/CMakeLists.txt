cmake_minimum_required(VERSION 3.9)

project(Shambackends CXX C)

set(Sources
        src/comm/details/CommunicationBufferImpl.cpp
        src/queues.cpp
        src/usmbuffer.cpp
        src/Device.cpp
    )

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shambackends target (Shared lib)")
add_library(shambackends SHARED ${Sources})
else()
message("-- adding shambackends target (object lib)")
add_library(shambackends OBJECT ${Sources})
endif()

add_library(shambackends::shambackends ALIAS shambackends)

target_link_libraries(shambackends PUBLIC shambase)
target_link_libraries(shambackends PUBLIC shamcomm)

target_include_directories(shambackends PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")

