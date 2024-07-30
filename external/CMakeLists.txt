add_subdirectory(pybind11)

add_subdirectory(fmt)


if(SHAMROCK_USE_NVTX)
    add_subdirectory(NVTX/c)
endif()


add_subdirectory(coala)
#find_package(pybind11 CONFIG REQUIRED)
