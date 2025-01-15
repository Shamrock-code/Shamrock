add_subdirectory(pybind11)

option(USE_SYSTEM_FMTLIB "use fmt lib from the host system" Off)

if(NOT USE_SYSTEM_FMTLIB)
    message(STATUS "Using git submodule fmtlib")
    add_subdirectory(fmt)
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
