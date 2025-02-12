function(_git_submod_is_empty directory)
    file(GLOB files RELATIVE "${directory}" "${directory}/*")
    if(NOT files)

        message(FATAL_ERROR
            "The git submodule '${directory}' is empty\n"
            "please do : git submodule update --init --recursive\n"
        )

    endif()
    message(STATUS "The subdirectory '${directory}' contains ${files}")
endfunction()

_git_submod_is_empty(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

add_subdirectory(pybind11)

option(USE_SYSTEM_FMTLIB "use fmt lib from the host system" Off)

if(NOT USE_SYSTEM_FMTLIB)
    message(STATUS "Using git submodule fmtlib")

    option(USE_MANUAL_FMTLIB "Bypass fmt cmake integration" Off)

    if(USE_MANUAL_FMTLIB)
        # Completely bypass fmt cmake integration
        # This is sketchy but allows the inclusion of fmt wihout ever having to compile it
        # this solved issue on latest macos
        # on Christiano's laptop (that were due to anaconda, of course ...)
        message(STATUS "You are bypassing fmt cmake integration use it at your own risks !")
        message(STATUS "Manual inclusion path ${CMAKE_CURRENT_LIST_DIR}/fmt/include")
        add_library(fmt-header-only INTERFACE)
        add_library(fmt::fmt-header-only ALIAS fmt-header-only)
        target_compile_definitions(fmt-header-only INTERFACE FMT_HEADER_ONLY=1)
        target_compile_features(fmt-header-only INTERFACE cxx_std_11)
        target_include_directories(fmt-header-only
            BEFORE INTERFACE
            ${CMAKE_CURRENT_LIST_DIR}/fmt/include
        )
    else()
        add_subdirectory(fmt)
    endif()
endif()

if(SHAMROCK_USE_NVTX)
    #include(NVTX/c/nvtxImportedTargets.cmake)
    add_subdirectory(NVTX/c)
endif()


if(SHAMROCK_EXTERNAL_JSON)
    find_package(nlohmann_json 3.11.3 REQUIRED)
else()
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    add_subdirectory(nlohmann_json)
endif()
