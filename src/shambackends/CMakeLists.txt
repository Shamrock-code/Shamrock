cmake_minimum_required(VERSION 3.9)

project(Shambackends CXX C)

message("-- adding shambackends target")
add_library(shambackends INTERFACE)

target_include_directories(shambackends INTERFACE
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")


# check that the wanted sycl backend is in the list
set(KNOWN_SHAMBACKENDS_USE "SYCL;KOKKOS")
if((NOT ${SHAMBACKENDS_USE} IN_LIST KNOWN_SHAMBACKENDS_USE) OR (NOT (DEFINED SHAMBACKENDS_USE)))
  message(FATAL_ERROR
    "No backends were selected for ShamBackends please select one "
    "-DSHAMBACKENDS_USE=[SYCL;KOKKOS]")
endif()

add_subdirectory(external)

if(${SHAMBACKENDS_USE} STREQUAL "SYCL")
    message(STATUS "adding sycl backend")
    target_compile_definitions(shambackends INTERFACE "SHAMBACKENDS_USE_SYCL")
elseif(${SHAMBACKENDS_USE} STREQUAL "KOKKOS")
    message(STATUS "adding kokkos backend")
    target_link_libraries(shambackends INTERFACE Kokkos::kokkos)
    target_compile_definitions(shambackends INTERFACE "SHAMBACKENDS_USE_KOKKOS")
endif()

set(SHAMBACKENDS_USE "${SHAMBACKENDS_USE}" CACHE STRING "Variable saying which backend is in use")
