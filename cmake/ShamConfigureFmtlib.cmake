## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- fmtlib section ----")

###############################################################################
### fmt
###############################################################################

option(SHAMROCK_EXTERNAL_FMTLIB "use fmt lib from the host system" Off)

# LEGACY variable
if(DEFINED USE_SYSTEM_FMTLIB)
    set(SHAMROCK_EXTERNAL_FMTLIB ${USE_SYSTEM_FMTLIB})
endif()

message(STATUS "SHAMROCK_EXTERNAL_FMTLIB : ${SHAMROCK_EXTERNAL_FMTLIB}")

if(NOT SHAMROCK_EXTERNAL_FMTLIB)
    message(STATUS "Using git submodule fmtlib")

    _check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/fmt 8303d140)

    option(USE_MANUAL_FMTLIB "Bypass fmt cmake integration" Off)

    if(USE_MANUAL_FMTLIB)
        # Completely bypass fmt cmake integration
        # This is sketchy but allows the inclusion of fmt wihout ever having to compile it
        # this solved issue on latest macos
        # on Christiano's laptop (that were due to anaconda, of course ...)
        message(STATUS "You are bypassing fmt cmake integration use it at your own risks !")
        message(STATUS "Manual inclusion path ${CMAKE_CURRENT_LIST_DIR}/external/fmt/include")
        add_library(fmt-header-only INTERFACE)
        add_library(fmt::fmt-header-only ALIAS fmt-header-only)
        target_compile_definitions(fmt-header-only INTERFACE FMT_HEADER_ONLY=1)
        target_compile_features(fmt-header-only INTERFACE cxx_std_11)
        target_include_directories(fmt-header-only
            BEFORE INTERFACE
            ${CMAKE_CURRENT_LIST_DIR}external//fmt/include
        )
    else()
        add_subdirectory(external/fmt)
    endif()

    # I got many clang-tidy warning because of those headers, so now they are system headers
    # just so that clang-tidy shut up

    # Override include directories property as system includes instead to avoid warnings
    # From https://stackoverflow.com/questions/64064157/is-there-a-way-to-get-isystem-for-fetchcontent-targets
    get_target_property(fmtlib_IID fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(fmt-header-only PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${fmtlib_IID}")

else()
    message(STATUS "Using system fmtlib")
    find_package(fmt REQUIRED)
endif()
