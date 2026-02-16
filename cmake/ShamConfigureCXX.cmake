## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- c++ config section ----")

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_MARCHNATIVE)
check_cxx_compiler_flag("-pedantic-errors" COMPILER_SUPPORT_PEDANTIC)
check_cxx_compiler_flag("-fcolor-diagnostics" COMPILER_SUPPORT_COLOR_DIAGNOSTIC)
check_cxx_compiler_flag("-Werror=return-type" COMPILER_SUPPORT_ERROR_RETURN_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")# -fsanitize=address")# -Wall -Wextra") #
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")#-DNDEBUG ")#-Wall -Wextra -Wunknown-cuda-version -Wno-linker-warnings")

if(COMPILER_SUPPORT_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
endif()

if(COMPILER_SUPPORT_COLOR_DIAGNOSTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
endif()

if(COMPILER_SUPPORT_ERROR_RETURN_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif()

if(COMPILER_SUPPORT_MARCHNATIVE)
    option(CXX_FLAG_ARCH_NATIVE "Use -march=native flag" On)
    if(CXX_FLAG_ARCH_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
endif()


check_cxx_source_compiles("
    #include <valarray>
    int main(){}
    "
    CXX_VALARRAY_COMPILE)

# this is a check used on systems with GCC 10.2.1-6 20210110
# because of a mismatch between valarray declaration and header
# bug was created by this https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103022
# see : https://bugs.mageia.org/show_bug.cgi?id=30658
if(NOT CXX_VALARRAY_COMPILE)

    check_cxx_source_compiles("
        #include <utility>
        #include <type_traits>
        #include <algorithm>
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored \"-Wkeyword-macro\"
        #define noexcept
        #include <valarray>
        #undef noexcept
        #pragma GCC diagnostic pop
        int main(){}
        "
        CXX_VALARRAY_COMPILE_NOEXCEPT)


    if(CXX_VALARRAY_COMPILE_NOEXCEPT)
        message(STATUS "Enable noexcept fix for valarray (#define SHAMROCK_VALARRAY_FIX)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_VALARRAY_FIX")
    endif()

endif()


message( STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message( STATUS "CMAKE_CXX_FLAGS_DEBUG : ${CMAKE_CXX_FLAGS_DEBUG}")
message( STATUS "CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")

######################
# add build types
######################

include(ShamBuildAsan)
include(ShamBuildUBsan)
include(ShamBuildMsan)
include(ShamBuildTsan)
include(ShamBuildCoverage)

set(ValidShamBuildType "Debug" "Release" "ASAN" "UBSAN" "MSAN" "TSAN" "COVERAGE")
if(NOT CMAKE_BUILD_TYPE)
    #set default build to release
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Cmake build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${ValidShamBuildType})
endif()
if(NOT "${CMAKE_BUILD_TYPE}" IN_LIST ValidShamBuildType)
    message(FATAL_ERROR
        "The required build type in unknown -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}. "
        "please use a build type in the following list (case-sensitive) "
        "${ValidShamBuildType}")
endif()
message(STATUS "current build type : CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${ValidShamBuildType})
