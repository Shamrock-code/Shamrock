cmake_minimum_required(VERSION 3.9)

project(Shambackends CXX C)

message("-- adding shambackends target")
add_library(shambackends INTERFACE)

target_include_directories(shambackends INTERFACE
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")


add_subdirectory(external)
if(${SHAMBACKENDS_USE} STREQUAL "SYCL")
    message(STATUS "adding sycl backend")
elseif(${SHAMBACKENDS_USE} STREQUAL "KOKKOS")
    message(STATUS "adding kokkos backend")
    target_link_libraries(shambackends)
else()
    message(FATAL_ERROR
        "No backends were selected for ShamBackends please select one "
        "-DSYCL_IMPLEMENTATION=[SYCL;KOKKOS]")
endif()

set(SHAMBACKENDS_USE "${SHAMBACKENDS_USE}" CACHE STRING "Variable saying which backend is in use")
