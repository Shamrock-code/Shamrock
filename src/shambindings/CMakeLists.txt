## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

cmake_minimum_required(VERSION 3.9...3.10)

project(Shambindings CXX C)

# Whenever this glob's value changes, cmake will rerun and update the build with the
# new/removed files.
file(GLOB_RECURSE glob_bindings_sources CONFIGURE_DEPENDS "*.cpp")

set(Sources
    ${glob_bindings_sources}
)

if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shambindings target (Shared lib)")
add_library(shambindings SHARED ${Sources})
else()
message("-- adding shambindings target (object lib)")
add_library(shambindings OBJECT ${Sources})
endif()

add_library(shambindings::shambindings ALIAS shambindings)

target_link_libraries(shambindings PUBLIC shambase)
target_link_libraries(shambindings PUBLIC shamcmdopt)
target_link_libraries(shambindings PUBLIC pybind11::pybind11)

target_include_directories(shambindings PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")



#####################################
# python sys.path config
#####################################

if(NOT DEFINED PYTHON_EXECUTABLE)
    message(FATAL_ERROR "This should have been defined in the main CmakeLists.txt")
endif()
message(STATUS "PYTHON_EXECUTABLE : ${PYTHON_EXECUTABLE}")

execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "-c" "import sys;print(sys.path, end= '')"
  RESULT_VARIABLE __PYTHON_SYS_PATH_RESULT
  OUTPUT_VARIABLE __PYTHON_SYS_PATH_OUT
  ERROR_VARIABLE __PYTHON_SYS_PATH_ERROR)

  #message(status " PYTHON_EXECUTABLE        : ${PYTHON_EXECUTABLE}")
  #message(status " __PYTHON_SYS_PATH_RESULT : ${__PYTHON_SYS_PATH_RESULT}")
  #message(status " __PYTHON_SYS_PATH_OUT    : ${__PYTHON_SYS_PATH_OUT}")
  #message(status " __PYTHON_SYS_PATH_ERROR  : ${__PYTHON_SYS_PATH_ERROR}")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/configure_time_py_sys_path.cpp
  "
  const char* configure_time_py_sys_path() { return \"${__PYTHON_SYS_PATH_OUT}\"; }
  const char* configure_time_py_executable() { return \"${PYTHON_EXECUTABLE}\"; }
  "
)

target_sources(shambindings PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/configure_time_py_sys_path.cpp)


#####################################
# pylib paths config
#####################################

set(PYLIB_PATHS
  "${CMAKE_INSTALL_PYTHONDIR}"
  "${CMAKE_SOURCE_DIR}/src/pylib"
)

# Escape for C++ string literals
string(REPLACE "\\" "\\\\" PYLIB_PATHS "${PYLIB_PATHS}")
string(REPLACE "\"" "\\\"" PYLIB_PATHS "${PYLIB_PATHS}")
string(JOIN ",\n        " PYLIB_PATHS_CPP "\"${PYLIB_PATHS}\"")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/configure_time_pylib_paths.cpp" "
  const char* configure_time_pylib_paths() { return \"${PYLIB_PATHS}\"; }
")

target_sources(shambindings PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/configure_time_pylib_paths.cpp")


#####################################
# run_ipython.py config
#####################################

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/run_ipython.py __CAT_RUNIPYTHON_OUT)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/run_ipython_src.cpp
  "const char* run_ipython_src() { return R\"=(\n${__CAT_RUNIPYTHON_OUT}\n)=\"; }"
)

target_sources(shambindings PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/run_ipython_src.cpp)
