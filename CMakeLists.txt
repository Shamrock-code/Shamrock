## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
## Licensed under CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

################################
# Cmake config
################################
cmake_minimum_required(VERSION 3.9)

# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    )

option(EXPORT_COMPILE_COMMANDS "export the compile_commands.json" ON)

# Necessary for clangd to work properly
if(EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

################################
# Shamrock configuration
################################  
project(Shamrock VERSION 2023.0)


set(SHAMROCK_ENABLE_BACKEND "SYCL" CACHE STRING "Shamrock backend used")
set_property(CACHE SHAMROCK_ENABLE_BACKEND PROPERTY STRINGS SYCL)








######################
#Project config
######################




option(SHAMROCK_USE_SHARED_LIB "use shared libraries" On)
option(SHAMROCK_USE_PROFILING "use custom profiling tooling" On)
option(SHAMROCK_USE_PCH "use precompiled headers" Off)
option(SHAMROCK_USE_NVTX "use nvtx tooling" On)



set(Default_comp "${CMAKE_CXX_COMPILER}")

LIST(APPEND CMAKE_SYSTEM_PROGRAM_PATH  "${COMP_ROOT_DIR}/bin")
LIST(APPEND CMAKE_SYSTEM_LIBRARY_PATH  "${COMP_ROOT_DIR}/bin/lib")




######################
#checking compiler flag support 
######################
include(ShamConfigureCXX)



######################
# Compiler selection
######################
include(ShamConfigureSYCL)
if(${SYCL_COMPILER} STREQUAL "DPCPP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_DPCPP -Wno-unknown-cuda-version")
elseif(${SYCL_COMPILER} STREQUAL "OPENSYCL")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_OPENSYCL")
elseif(${SYCL_COMPILER} STREQUAL "OTHER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSYCL_COMP_SYCLUNKNOWN")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${COMP_ROOT_DIR}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${COMP_ROOT_DIR}/include/sycl")




enable_language(C)
enable_language(CXX)


######################
#shamrock impl control
######################
if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR_ROUND")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")


if(SHAMROCK_USE_PROFILING)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_PROFILING")
endif()

if(SHAMROCK_USE_NVTX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_USE_NVTX")
endif()

######################
# Add external libraries
######################
set(MPI_CXX_SKIP_MPICXX true)
find_package(MPI REQUIRED COMPONENTS C)

message(STATUS "MPI include dir : ${MPI_C_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOMPI_SKIP_MPICXX")

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(external)

if("${BUILD_PYLIB}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_LIB_BUILD")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_EXECUTABLE_BUILD")
endif()

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


message("-- Shamrock config : ")
message("     SHAMROCK_LOOP_DEFAULT : ${SHAMROCK_LOOP_DEFAULT}")
message("     SHAMROCK_LOOP_GSIZE : ${SHAMROCK_LOOP_GSIZE}")
message("     SHAMROCK_USE_PROFILING : ${SHAMROCK_USE_PROFILING}")
message("     SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
message("     SHAMROCK_USE_PCH : ${SHAMROCK_USE_PCH}")
message("     SHAMROCK_USE_SHARED_LIB : ${SHAMROCK_USE_SHARED_LIB}")
message("     SYCL_COMPILER : ${SYCL_COMPILER}")
message("     CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message("     CMAKE_CXX_COMPILER : ${CMAKE_CXX_COMPILER}")


########################################################################################
# Targets ##############################################################################
########################################################################################

add_subdirectory(src/shamunits)

######################
# Shamalgs
######################
if(SHAMROCK_USE_SHARED_LIB)
message("-- adding shamalgs target (Shared lib)")
add_library(shamalgs SHARED "")
else()
message("-- adding shamalgs target (no lib)")
add_library(shamalgs OBJECT "")
endif()

target_include_directories(shamalgs PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamalgs PRIVATE "<sycl/sycl.hpp>")
endif()

target_link_libraries(shamalgs PUBLIC MPI::MPI_C)
target_link_libraries(shamalgs PUBLIC fmt::fmt-header-only)

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamalgs PUBLIC nvtx3-cpp)
endif()

target_link_libraries(shamalgs PUBLIC shamunits)







######################
# Shamrock Lib
######################

message("-- adding shamrock_lib target")
add_library(shamrock_lib OBJECT src/version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock_lib PRIVATE "<sycl/sycl.hpp>")
endif()

target_link_libraries(shamrock_lib PUBLIC MPI::MPI_C)
target_link_libraries(shamrock_lib PUBLIC fmt::fmt)

if(SHAMROCK_USE_NVTX)
    target_link_libraries(shamrock_lib PUBLIC nvtx3-cpp)
endif()

if("${BUILD_PYLIB}")
    target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
    target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()


target_link_libraries(shamrock_lib PUBLIC shamunits)







if("${BUILD_SIM}")

    message("-- adding shamrock target")


    if("${BUILD_PYLIB}")
    pybind11_add_module(shamrock  )
    target_link_libraries(shamrock PUBLIC pybind11::module)
    else()
    add_executable(shamrock src/main.cpp )
    target_link_libraries(shamrock PUBLIC pybind11::embed)
    endif()
    
    target_link_libraries(shamrock PUBLIC shamrock_lib)
    target_link_libraries(shamrock PUBLIC shamalgs)
    
    
    target_link_libraries(shamrock PUBLIC MPI::MPI_C)
    target_link_libraries(shamrock PUBLIC fmt::fmt)
    target_link_libraries(shamrock PUBLIC shamunits)

    

    if(SHAMROCK_USE_NVTX)
        target_link_libraries(shamrock PUBLIC nvtx3-cpp)
    endif()



    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock REUSE_FROM shamrock_lib)
    endif()

    


    target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/external")

    #do lib build instead if building the pylib


endif()


if("${BUILD_TEST}")

    message("-- adding shamrock_test target")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test src/main_test.cpp  )
    target_link_libraries(shamrock_test PUBLIC MPI::MPI_C)

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)

    target_link_libraries(shamrock_test PUBLIC shamrock_lib)
    target_link_libraries(shamrock_test PUBLIC shamalgs)
    target_link_libraries(shamrock_test PUBLIC shamunits)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")

    target_link_libraries(shamrock_test PUBLIC fmt::fmt)

    if(SHAMROCK_USE_NVTX)
        target_link_libraries(shamrock_test PUBLIC nvtx3-cpp)
    endif()



    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock_test REUSE_FROM shamrock_lib)
    endif()

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()

add_subdirectory(src)