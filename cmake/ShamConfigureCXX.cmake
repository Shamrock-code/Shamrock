include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_MARCHNATIVE)
check_cxx_compiler_flag("-pedantic-errors" COMPILER_SUPPORT_PEDANTIC)
check_cxx_compiler_flag("-fcolor-diagnostics" COMPILER_SUPPORT_COLOR_DIAGNOSTIC)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type") 
set(CMAKE_CXX_FLAGS_DEBUG "-g")# -fsanitize=address")# -Wall -Wextra") #
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")#-DNDEBUG ")#-Wall -Wextra -Wunknown-cuda-version -Wno-linker-warnings")

if(COMPILER_SUPPORT_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors") 
endif()

if(COMPILER_SUPPORT_COLOR_DIAGNOSTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics") 
endif()

if(COMPILER_SUPPORT_MARCHNATIVE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native") 
endif()