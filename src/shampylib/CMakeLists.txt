cmake_minimum_required(VERSION 3.9)

project(Shampylib CXX C)

# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE shampylib_srcs CONFIGURE_DEPENDS "*.cpp")

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shampylib target (Shared lib)")
add_library(shampylib SHARED ${shampylib_srcs})
else()
message("-- adding shampylib target (object lib)")
add_library(shampylib OBJECT ${shampylib_srcs})
endif()

add_library(shampylib::shampylib ALIAS shampylib)

target_link_libraries(shampylib PUBLIC shambackends)
target_link_libraries(shampylib PUBLIC shammath)
target_link_libraries(shampylib PUBLIC shamphys)
target_link_libraries(shampylib PUBLIC shamalgs)
target_link_libraries(shampylib PUBLIC shamsys)
target_link_libraries(shampylib PUBLIC shamlib)
target_link_libraries(shampylib PUBLIC shambindings)
target_link_libraries(shampylib PUBLIC shamtree)

target_include_directories(shampylib PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")
