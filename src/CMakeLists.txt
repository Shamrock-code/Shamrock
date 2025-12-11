## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

set(as_subproject
  shamunits
  shambase
  shamcomm
  shambackends
  shammath
  shamalgs
  shamsys
  shamphys
  shamcmdopt
  shambindings
  shamtree
  shamlib
  shampylib

  shammodels_common
  shammodels_nbody
  shammodels_zeus
  shammodels_ramses
  shammodels_sph
  )

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
add_subdirectory(shambindings)
add_subdirectory(shamcmdopt)
add_subdirectory(shamcomm)
add_subdirectory(shambackends)
add_subdirectory(shammath)
add_subdirectory(shamalgs)
add_subdirectory(shamphys)
add_subdirectory(shamsys)
add_subdirectory(shamtree)
add_subdirectory(shamrock)
add_subdirectory(shampylib)

add_subdirectory(shammodels/common)
add_subdirectory(shammodels/nbody)
add_subdirectory(shammodels/zeus)
add_subdirectory(shammodels/ramses)
add_subdirectory(shammodels/sph)



######################
# Shamrock lib & exe targets
######################



message("-- adding shamrock target")


pybind11_add_module(shamrock_pylib main_lib.cpp)
target_link_libraries(shamrock_pylib PUBLIC pybind11::module)
set_property(TARGET shamrock_pylib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_executable(shamrock_exe main.cpp)
target_link_libraries(shamrock_exe PUBLIC pybind11::embed)

function(target_link_lib_and_install tgt)
    target_link_libraries(${tgt} PUBLIC nlohmann_json::nlohmann_json)

    target_link_libraries(${tgt} PUBLIC ${as_subproject})

    if(SHAMROCK_USE_PCH)
        target_precompile_headers(${tgt} REUSE_FROM shamrock_lib)
    endif()

    target_include_directories(${tgt} PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(${tgt} PUBLIC "${CMAKE_SOURCE_DIR}/external")

endfunction()

target_link_lib_and_install(shamrock_exe)
target_link_lib_and_install(shamrock_pylib)

set_property(TARGET shamrock_exe PROPERTY OUTPUT_NAME shamrock)
set_property(TARGET shamrock_pylib PROPERTY OUTPUT_NAME pyshamrock)


########################################################################################
# Shamrock install target
########################################################################################

# Installing the Shamrock and its libraries
install(TARGETS shamrock_exe ${as_subproject}
        EXPORT "shamrock-config"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        PRIVATE_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")


if(CMAKE_INSTALL_PYTHONDIR)

  # For python, we only install the bindings and the __init__.py file
  # running `̀import shamrock` in python will load the __init__.py file
  # which in turn will load shamrock_pylib
  # Two options there
  # 1 - The shamrock libraries `${as_subproject}` are not in path and you get
  # ImportError: libshampylib.so: cannot open shared object file: No such file or directory
  # 2 - The shamrock libraries `${as_subproject}` are in path and it works,
  #     you can check that by running, on linux, `ldd shamrock/shamrock.cpython-313-x86_64-linux-gnu.so`
  #     in ${CMAKE_INSTALL_PYTHONDIR}

  message(STATUS "Installing python bindings in ${CMAKE_INSTALL_PYTHONDIR}")

  # Set the install dir for the shamrock package
  set(SHAMROCK_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PYTHONDIR}/shamrock")

  # Copy content of pylib/shamrock to the install directory
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/src/pylib/shamrock"
          DESTINATION "${CMAKE_INSTALL_PYTHONDIR}")

  # Install the Python module shared library
  install(TARGETS shamrock_pylib
          LIBRARY DESTINATION "${SHAMROCK_PYTHON_INSTALL_DIR}")
endif()

########################################################################################
# Shamrock test target
########################################################################################

if("${BUILD_TEST}")

    message("-- adding shamrock_test target")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test main_test.cpp  )

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)
    target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

    target_link_libraries(shamrock_test PUBLIC ${as_subproject})


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


add_subdirectory(shamtest)


#add_subdirectory(runscript)

if("${BUILD_TEST}")
  add_subdirectory(tests)
endif()
