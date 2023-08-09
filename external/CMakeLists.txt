add_subdirectory(pybind11)

add_subdirectory(fmt EXCLUDE_FROM_ALL)


if(SHAMROCK_USE_NVTX)
    #include(NVTX/c/nvtxImportedTargets.cmake)
    add_subdirectory(NVTX/c)
endif()


#find_package(pybind11 CONFIG REQUIRED)