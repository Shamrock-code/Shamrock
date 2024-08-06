add_subdirectory(pybind11)

add_subdirectory(fmt)


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
