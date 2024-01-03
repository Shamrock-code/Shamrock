######################
# generate version.cpp
######################
#hack to rerun the version generator if ninja/make all
#cf https://stackoverflow.com/questions/13920072/how-to-always-run-command-when-building-regardless-of-any-dependency
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp 
           ${CMAKE_CURRENT_SOURCE_DIR}/version__.cpp 
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/../buildbot/make_version_file.py ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp "${CMAKE_CXX_FLAGS}"
  )





set(as_subproject shamunits shambase shamcomm)

# This macro originates from Daniel Pfeifer's talk at C++Now 2017:
# https://www.youtube.com/watch?v=bsXLMQ6WgIk
macro(find_package)
  if(NOT "${ARGV0}" IN_LIST as_subproject)
    message(STATUS "using native find_package for ${ARGV0}: ${ARGV}")
    _find_package(${ARGV})
  endif()
endmacro()

add_subdirectory(shamunits)
add_subdirectory(shambase)
add_subdirectory(shamcomm)

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

target_link_libraries(shamalgs PUBLIC shamunits)
target_link_libraries(shamalgs PUBLIC shambase)
target_link_libraries(shamalgs PUBLIC shamcomm)






######################
# Shamrock Lib
######################

message("-- adding shamrock_lib target")
add_library(shamrock_lib OBJECT version.cpp)
target_include_directories(shamrock_lib PUBLIC "${CMAKE_SOURCE_DIR}/src")

if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock_lib PRIVATE "<sycl/sycl.hpp>")
endif()



if("${BUILD_PYLIB}")
    target_link_libraries(shamrock_lib PUBLIC pybind11::module)
else()
    target_link_libraries(shamrock_lib PUBLIC pybind11::embed)
endif()


target_link_libraries(shamrock_lib PUBLIC shamunits)
target_link_libraries(shamrock_lib PUBLIC shambase)
target_link_libraries(shamrock_lib PUBLIC shamcomm)






message("-- adding shamrock target")


if("${BUILD_PYLIB}")
pybind11_add_module(shamrock  )
target_link_libraries(shamrock PUBLIC pybind11::module)
else()
add_executable(shamrock main.cpp )
target_link_libraries(shamrock PUBLIC pybind11::embed)
endif()

target_link_libraries(shamrock PUBLIC shamrock_lib)
target_link_libraries(shamrock PUBLIC shamalgs)


target_link_libraries(shamrock PUBLIC shamunits)
target_link_libraries(shamrock PUBLIC shambase)
target_link_libraries(shamrock PUBLIC shamcomm)




if(SHAMROCK_USE_PCH)
target_precompile_headers(shamrock REUSE_FROM shamrock_lib)
endif()




target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/src")
#target_include_directories(shamrock PUBLIC "${CMAKE_SOURCE_DIR}/external")

#do lib build instead if building the pylib




if("${BUILD_TEST}")

    message("-- adding shamrock_test target")

    #add_executable(shamrock_test main_test.cpp version.cpp)
    add_executable(shamrock_test main_test.cpp  )

    target_link_libraries(shamrock_test PUBLIC pybind11::embed)

    target_link_libraries(shamrock_test PUBLIC shamrock_lib)
    target_link_libraries(shamrock_test PUBLIC shamalgs)
    target_link_libraries(shamrock_test PUBLIC shamunits)
    target_link_libraries(shamrock_test PUBLIC shambase)
    target_link_libraries(shamrock_test PUBLIC shamcomm)

    
    target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/src")
    #target_include_directories(shamrock_test PUBLIC "${CMAKE_SOURCE_DIR}/external")


    if(SHAMROCK_USE_PCH)
    target_precompile_headers(shamrock_test REUSE_FROM shamrock_lib)
    endif()

    #find_package(nlohmann_json 3.2.0 REQUIRED)
    #target_link_libraries(shamrock_test PUBLIC nlohmann_json::nlohmann_json)

endif()







add_subdirectory(shamsys)
add_subdirectory(shamtest)
add_subdirectory(shambindings)

add_subdirectory(shamalgs)
add_subdirectory(shammath)

add_subdirectory(shamrock)
add_subdirectory(shammodels)
add_subdirectory(shambackends)

#add_subdirectory(runscript)

add_subdirectory(tests)
