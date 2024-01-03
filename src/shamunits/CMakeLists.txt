cmake_minimum_required(VERSION 3.9)

project(Shamunits CXX C)

message("-- adding shamunits target (interface lib)")
add_library(shamunits INTERFACE)
add_library(shamunits::shamunits ALIAS shamunits)

target_include_directories(shamunits INTERFACE
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
