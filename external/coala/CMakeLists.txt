cmake_minimum_required(VERSION 3.9)

project(Coala CXX C)

message("-- adding coala target")
add_library(coala INTERFACE)

target_include_directories(coala INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

