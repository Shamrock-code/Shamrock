
if("${SHAMROCK_ENABLE_BACKEND}" STREQUAL "SYCL")
    add_subdirectory(sycl)
elseif("${SHAMROCK_ENABLE_BACKEND}" STREQUAL "KOKKOS")
    add_subdirectory(kokkos)
endif()
