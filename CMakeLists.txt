######################################
# Cmake build configuration for shamrock
######################################

cmake_minimum_required(VERSION 3.9)


project(Shamrock CXX C)


# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)



######################
#Project config
######################

set(SYCL_COMPILER "OTHER" CACHE STRING "Sycl compiler used")
set_property(CACHE SYCL_COMPILER PROPERTY STRINGS DPCPP OPENSYCL OTHER)

set(SHAMROCK_LOOP_DEFAULT "PARRALEL_FOR_ROUND" CACHE STRING "Default loop mode in shamrock")
set_property(CACHE SHAMROCK_LOOP_DEFAULT PROPERTY STRINGS PARRALEL_FOR PARRALEL_FOR_ROUND ND_RANGE)

set(SHAMROCK_LOOP_GSIZE 256 CACHE STRING "Default group size in shamrock")

option(EXPORT_COMPILE_COMMANDS "export the compile_commands.json" ON)
option(SHAMROCK_USE_NVTX "use nvtx tooling" On)
option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
option(SHAMROCK_USE_PCH "use precompiled headers" Off)

message("-- Shamrock config : ")
message("     SHAMROCK_LOOP_DEFAULT : ${SHAMROCK_LOOP_DEFAULT}")
message("     SHAMROCK_LOOP_GSIZE : ${SHAMROCK_LOOP_GSIZE}")
message("     SHAMROCK_USE_PROFILING : ${SHAMROCK_USE_PROFILING}")
message("     SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
message("     SHAMROCK_USE_PCH : ${SHAMROCK_USE_PCH}")
message("     SYCL_COMPILER : ${SYCL_COMPILER}")
message("     CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message("     CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")



# Necessary for clangd to work properly
if(EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(Default_comp "${CMAKE_CXX_COMPILER}")

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
if(${SYCL_COMPILER} STREQUAL "DPCPP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_DPCPP -Wno-unknown-cuda-version")
elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_OPENSYCL")
elseif(${SYCL_COMPILER} STREQUAL "OTHER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_SYCLUNKNOWN")
endif()


######################
#shamrock impl control
######################
if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  add_compile_definitions("SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR_ROUND")
  add_compile_definitions("SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  add_compile_definitions("SHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

add_compile_definitions("SHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")



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
# generate version.cpp
######################
#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/src/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/buildbot/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp "${CMAKE_CXX_FLAGS}"
  )



########################################################################################
# Targets ##############################################################################
########################################################################################

######################
# Shamalgs
######################
add_library(shamalgs SHARED src/version.cpp)
target_include_directories(shamalgs PUBLIC "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(shamalgs PUBLIC MPI::MPI_C)
target_link_libraries(shamalgs PUBLIC fmt::fmt-header-only)

if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamalgs PRIVATE "<sycl/sycl.hpp>")
endif()

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamalgs PUBLIC nvtx3-cpp)
    target_compile_definitions(shamalgs PUBLIC SHAMROCK_USE_NVTX)
endif()

if(SHAMROCK_USE_PROFILING)
    target_compile_definitions(shamalgs PUBLIC SHAMROCK_USE_PROFILING)
endif()







######################
# Shamrock Lib
######################

add_library(shamrock_lib OBJECT src/version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(shamrock_lib PUBLIC MPI::MPI_C)
target_link_libraries(shamrock_lib PUBLIC fmt::fmt)

if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock_lib PRIVATE "<sycl/sycl.hpp>")
endif()

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamrock_lib PUBLIC nvtx3-cpp)
    target_compile_definitions(shamrock_lib PUBLIC SHAMROCK_USE_NVTX)
endif()

if(SHAMROCK_USE_PROFILING)
    target_compile_definitions(shamrock_lib PUBLIC SHAMROCK_USE_PROFILING)
endif()

if("${BUILD_PYLIB}")
    target_compile_definitions(shamrock_lib PUBLIC LIB_BUILD)
    target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
    target_compile_definitions(shamrock_lib PUBLIC EXECUTABLE_BUILD)
    target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()






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
    target_link_libraries(shamrock PUBLIC shamalgs)
    
    
    target_link_libraries(shamrock PUBLIC MPI::MPI_C)
    target_link_libraries(shamrock PUBLIC fmt::fmt)

    if(SHAMROCK_USE_NVTX)
        target_link_libraries(shamrock PUBLIC nvtx3-cpp)
        target_compile_definitions(shamrock PUBLIC SHAMROCK_USE_NVTX)
    endif()


    if(SHAMROCK_USE_PROFILING)
    target_compile_definitions(shamrock PUBLIC SHAMROCK_USE_PROFILING)
    endif()

    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock REUSE_FROM shamrock_lib)
    endif()

    


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
    target_link_libraries(shamrock_test PUBLIC shamalgs)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")

    target_compile_definitions(shamrock_test PUBLIC EXECUTABLE_BUILD)
    target_link_libraries(shamrock_test PUBLIC fmt::fmt)

    if(SHAMROCK_USE_NVTX)
        target_link_libraries(shamrock_test PUBLIC nvtx3-cpp)
        target_compile_definitions(shamrock_test PUBLIC SHAMROCK_USE_NVTX)
    endif()


    if(SHAMROCK_USE_PROFILING)
    target_compile_definitions(shamrock_test PUBLIC SHAMROCK_USE_PROFILING)
    endif()

    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock_test REUSE_FROM shamrock_lib)
    endif()

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()

add_subdirectory(src)