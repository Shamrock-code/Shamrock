add_subdirectory(pybind11)

add_subdirectory(fmt)

message("-- SHAMROCK_USE_NVTX : ${SHAMROCK_USE_NVTX}")
if(SHAMROCK_USE_NVTX)
    add_subdirectory(NVTX/c)
endif()


#find_package(pybind11 CONFIG REQUIRED)