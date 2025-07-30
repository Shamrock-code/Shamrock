## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

# TODO rename to shamrock_lib when the old one will be removed

project(Shamlib CXX C)

######################
# generate version.cpp
######################
#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
if(${CMAKE_VERSION} VERSION_GREATER 3.27)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/version__.cpp
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../buildbot/make_version_file.py ${CMAKE_CURRENT_BINARY_DIR}/version.cpp "${CMAKE_CXX_FLAGS}" "${SHAMROCK_VERSION_STRING}"
    DEPENDS_EXPLICIT_ONLY
  )
else()
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/version__.cpp
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../../buildbot/make_version_file.py ${CMAKE_CURRENT_BINARY_DIR}/version.cpp "${CMAKE_CXX_FLAGS}" "${SHAMROCK_VERSION_STRING}"
  )
endif()


# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE shamlib_srcs CONFIGURE_DEPENDS "*.cpp")

if(SHAMROCK_USE_SHARED_LIB)
  message("-- adding shamlib target (Shared lib)")
  add_library(shamlib SHARED ${CMAKE_CURRENT_BINARY_DIR}/version.cpp ${shamlib_srcs})
else()
  message("-- adding shamlib target (object lib)")
  add_library(shamlib OBJECT ${CMAKE_CURRENT_BINARY_DIR}/version.cpp ${shamlib_srcs})
endif()

add_library(shamlib::shamlib ALIAS shamlib)


target_link_libraries(shamlib PUBLIC shambackends)
target_link_libraries(shamlib PUBLIC shammath)
target_link_libraries(shamlib PUBLIC shamphys)
target_link_libraries(shamlib PUBLIC shamalgs)
target_link_libraries(shamlib PUBLIC shamsys)
target_link_libraries(shamlib PUBLIC shambindings)
target_link_libraries(shamlib PUBLIC shamtree)
target_link_libraries(shamlib PUBLIC nlohmann_json::nlohmann_json)

target_include_directories(shamlib PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
