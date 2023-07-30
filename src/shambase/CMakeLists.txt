cmake_minimum_required(VERSION 3.9)

project(Shambase CXX C)

# Necessary to link a subfolder library into another subfolder
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)

add_library(shambase OBJECT exception.cpp
stacktrace.cpp)

if(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR")
  target_compile_definitions(shambase PUBLIC "SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "PARRALEL_FOR_ROUND")
  target_compile_definitions(shambase PUBLIC "SHAMROCK_LOOP_DEFAULT_PARRALEL_FOR_ROUND")
elseif(${SHAMROCK_LOOP_DEFAULT} STREQUAL "ND_RANGE")
  target_compile_definitions(shambase PUBLIC "SHAMROCK_LOOP_DEFAULT_ND_RANGE")
endif()

target_compile_definitions(shambase PUBLIC "SHAMROCK_LOOP_GSIZE=${SHAMROCK_LOOP_GSIZE}")

target_include_directories(shambase PUBLIC include)

