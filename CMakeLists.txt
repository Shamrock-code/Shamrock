######################################
# Cmake build configuration for shamrock
######################################

cmake_minimum_required(VERSION 3.9)
project(Shamrock CXX C)


# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)




set(SHAMROCK_LOOP_DEFAULT "ND_RANGE" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR ND_RANGE)

option(EXPORT_COMPILE_COMMANDS "export the compile_commands.json" ON)






# Necessary for clangd to work properly
if(EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(Default_comp "${CMAKE_CXX_COMPILER}")

message(STATUS "chosen compiler : ${SyCL_Compiler}")

LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${COMP_ROOT_DIR}/bin")
LIST(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${COMP_ROOT_DIR}/bin/lib")

######################
#checking compiler flag support 
######################
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_MARCHNATIVE)
check_cxx_compiler_flag("-pedantic-errors" COMPILER_SUPPORT_PEDANTIC)
check_cxx_compiler_flag("-fcolor-diagnostics" COMPILER_SUPPORT_COLOR_DIAGNOSTIC)

######################
# Add compilation flags depending upon the build type
######################
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type") 
set(CMAKE_CXX_FLAGS_DEBUG "-g")# -fsanitize=address")# -Wall -Wextra") #
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")#-DNDEBUG ")#-Wall -Wextra -Wunknown-cuda-version -Wno-linker-warnings")

if(COMPILER_SUPPORT_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors") 
endif()

if(COMPILER_SUPPORT_COLOR_DIAGNOSTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics") 
endif()

if(COMPILER_SUPPORT_MARCHNATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native") 
endif()



######################
# Compiler selection
######################
if("${SyCL_Compiler}" STREQUAL "DPCPP")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_DPCPP -Wno-unknown-cuda-version")

endif()


if("${SyCL_Compiler}" STREQUAL "OPENSYCL")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_OPENSYCL")

endif()


######################
#shamrock impl control
######################

if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  add_compile_definitions("SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  add_compile_definitions("SHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()



######################
# Add external libraries
######################
set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(external)


#for some reason it doesn't work with target_include_directories
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${COMP_ROOT_DIR}/include -isystem ${COMP_ROOT_DIR}/include/sycl -DOMPI_SKIP_MPICXX")

include_directories(Before System external/plf_nanotimer)

######################
# Add subdirectories
######################

#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/src/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/buildbot/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp "${CMAKE_CXX_FLAGS}"
  )

add_library(shamrock_lib OBJECT src/version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

if("${BUILD_PYLIB}")
target_compile_definitions(shamrock_lib PUBLIC LIB_BUILD)
target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
target_compile_definitions(shamrock_lib PUBLIC EXECUTABLE_BUILD)
target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()

target_link_libraries(shamrock_lib PUBLIC MPI::MPI_C)
target_link_libraries(shamrock_lib PUBLIC fmt::fmt)

if("${BUILD_SIM}")

    message(">>> adding sph target <<<")


    if("${BUILD_PYLIB}")
    pybind11_add_module(shamrock  )
    target_compile_definitions(shamrock PUBLIC LIB_BUILD)
    target_link_libraries(shamrock PUBLIC pybind11::module)
    else()
    add_executable(shamrock src/main.cpp )
    target_compile_definitions(shamrock PUBLIC EXECUTABLE_BUILD)
    target_link_libraries(shamrock PUBLIC pybind11::embed)
    endif()
    
    target_link_libraries(shamrock PUBLIC shamrock_lib)
    
    
    target_link_libraries(shamrock PUBLIC MPI::MPI_C)
    target_link_libraries(shamrock PUBLIC fmt::fmt)

    


    target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/external")

    #do lib build instead if building the pylib


endif()


if("${BUILD_TEST}")

    message(">>> adding test target <<<")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test src/main_test.cpp  )
    target_link_libraries(shamrock_test PUBLIC MPI::MPI_C)

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)

    target_link_libraries(shamrock_test PUBLIC shamrock_lib)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")

    target_compile_definitions(shamrock_test PUBLIC EXECUTABLE_BUILD)
    target_link_libraries(shamrock_test PUBLIC fmt::fmt)

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()

add_subdirectory(src)