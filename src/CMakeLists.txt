######################
# generate version.cpp
######################
#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../buildbot/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp "${CMAKE_CXX_FLAGS}"
  )





set(as_subproject 
  shamunits 
  shambase 
  shamcomm 
  shambackends 
  shammath 
  shamalgs 
  shamsys 
  shamphys 
  shamcmdopt)

# This macro originates from Daniel Pfeifer's talk at C++Now 2017:
# https://www.youtube.com/watch?v=bsXLMQ6WgIk
macro(find_package)
  if(NOT "${ARGV0}" IN_LIST as_subproject)
    message(STATUS "using native find_package for ${ARGV0}: ${ARGV}")
    _find_package(${ARGV})
  endif()
endmacro()

add_subdirectory(shamunits)
add_subdirectory(shambase)
add_subdirectory(shamcomm)
add_subdirectory(shambackends)
add_subdirectory(shammath)
add_subdirectory(shamalgs)
add_subdirectory(shamsys)
add_subdirectory(shamphys)
add_subdirectory(shamcmdopt)




######################
# Shamrock Lib
######################

message("-- adding shamrock_lib target")
add_library(shamrock_lib OBJECT version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock_lib PRIVATE "<sycl/sycl.hpp>")
endif()



if("${BUILD_PYLIB}")
    target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
    target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()


target_link_libraries(shamrock_lib PUBLIC shamunits)
target_link_libraries(shamrock_lib PUBLIC shambase)
target_link_libraries(shamrock_lib PUBLIC shamcomm)
target_link_libraries(shamrock_lib PUBLIC shambackends)
target_link_libraries(shamrock_lib PUBLIC shammath)
target_link_libraries(shamrock_lib PUBLIC shamalgs)
target_link_libraries(shamrock_lib PUBLIC shamsys)
target_link_libraries(shamrock_lib PUBLIC shamphys)






message("-- adding shamrock target")


if("${BUILD_PYLIB}")
pybind11_add_module(shamrock  )
target_link_libraries(shamrock PUBLIC pybind11::module)
else()
add_executable(shamrock main.cpp)
target_link_libraries(shamrock PUBLIC pybind11::embed)
endif()

target_link_libraries(shamrock PUBLIC shamrock_lib)
target_link_libraries(shamrock PUBLIC shamalgs)


target_link_libraries(shamrock PUBLIC shamunits)
target_link_libraries(shamrock PUBLIC shambase)
target_link_libraries(shamrock PUBLIC shamcomm)
target_link_libraries(shamrock PUBLIC shambackends)
target_link_libraries(shamrock PUBLIC shammath)
target_link_libraries(shamrock PUBLIC shamsys)
target_link_libraries(shamrock PUBLIC shamphys)




if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock REUSE_FROM shamrock_lib)
endif()




target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/src")
#target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/external")

#do lib build instead if building the pylib




if("${BUILD_TEST}")

    message("-- adding shamrock_test target")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test main_test.cpp  )

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)

    target_link_libraries(shamrock_test PUBLIC shamrock_lib)
    target_link_libraries(shamrock_test PUBLIC shamalgs)
    target_link_libraries(shamrock_test PUBLIC shamunits)
    target_link_libraries(shamrock_test PUBLIC shambase)
    target_link_libraries(shamrock_test PUBLIC shamcomm)
    target_link_libraries(shamrock_test PUBLIC shambackends)
    target_link_libraries(shamrock_test PUBLIC shammath)
    target_link_libraries(shamrock_test PUBLIC shamsys)
    target_link_libraries(shamrock_test PUBLIC shamphys)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")


    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock_test REUSE_FROM shamrock_lib)
    endif()

    ########################################################################################
    # CTest integration ####################################################################
    ########################################################################################

    enable_testing ()

    set(ctest_file_list "${CMAKE_CURRENT_BINARY_DIR}/cmake_tests.cmake")

    # add execution of a command just after linking of shamrock_test
    # this call generate the ctest cmake file
    add_custom_command(
      TARGET shamrock_test POST_BUILD
      BYPRODUCTS "${ctest_file_list}"
      COMMAND shamrock_test --gen-test-list "${ctest_file_list}" || true
      VERBATIM
    )

    # add the ctest cmake file to the list of tests
    set_property(DIRECTORY
      APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_file_list}"
    )

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()




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

target_sources(shamrock PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/ipython_run_src.cpp)

if("${BUILD_TEST}")
  target_sources(shamrock_test PRIVATE  ${CMAKE_CURRENT_BINARY_DIR}/ipython_run_src.cpp)
endif()






add_subdirectory(shamtest)
add_subdirectory(shambindings)


add_subdirectory(shamrock)
add_subdirectory(shammodels)

#add_subdirectory(runscript)

if("${BUILD_TEST}")
  add_subdirectory(tests)
endif()
