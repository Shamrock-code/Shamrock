cmake_minimum_required(VERSION 3.9)

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

if("${BUILD_PYLIB}")
    target_link_libraries(shambindings PUBLIC pybind11::module)
else()
    target_link_libraries(shambindings PUBLIC pybind11::embed)
endif()

target_include_directories(shambindings PUBLIC
"$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
"$<INSTALL_INTERFACE:include>")



execute_process(
  COMMAND
    "${Python_EXECUTABLE}" "-c" "
import sys;print(sys.path, end= '')
"
  RESULT_VARIABLE __PYTHON_SYS_PATH_RESULT
  OUTPUT_VARIABLE __PYTHON_SYS_PATH_OUT
  ERROR_VARIABLE __PYTHON_SYS_PATH_ERROR)

  #message(status " Python_EXECUTABLE        : ${Python_EXECUTABLE}")
  #message(status " __PYTHON_SYS_PATH_RESULT : ${__PYTHON_SYS_PATH_RESULT}")
  #message(status " __PYTHON_SYS_PATH_OUT    : ${__PYTHON_SYS_PATH_OUT}")
  #message(status " __PYTHON_SYS_PATH_ERROR  : ${__PYTHON_SYS_PATH_ERROR}")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ipython_run_src.cpp
  "const char* ipython_run_src() { return \"${__PYTHON_SYS_PATH_OUT}\"; }"
)

target_sources(shambindings PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/ipython_run_src.cpp)
