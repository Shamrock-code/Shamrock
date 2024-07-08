cmake_minimum_required(VERSION 3.9)

project(Shamphys CXX C)

set(Sources
    src/HydroSoundwave.cpp
    src/SodTube.cpp
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamphys target (Shared lib)")
add_library(shamphys SHARED ${Sources})
else()
message("-- adding shamphys target (object lib)")
add_library(shamphys OBJECT ${Sources})
endif()

add_library(shamphys::shamphys ALIAS shamphys)

target_link_libraries(shamphys PUBLIC shambackends)
target_link_libraries(shamphys PUBLIC shammath)
target_link_libraries(shamphys PUBLIC shamunits)

target_include_directories(shamphys PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
