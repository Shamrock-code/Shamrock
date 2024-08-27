include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_MARCHNATIVE)
check_cxx_compiler_flag("-pedantic-errors" COMPILER_SUPPORT_PEDANTIC)
check_cxx_compiler_flag("-fcolor-diagnostics" COMPILER_SUPPORT_COLOR_DIAGNOSTIC)

check_cxx_compiler_flag("-Werror=return-type" COMPILER_SUPPORT_ERROR_RETURN_TYPE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g")# -fsanitize=address")# -Wall -Wextra") #
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")#-DNDEBUG ")#-Wall -Wextra -Wunknown-cuda-version -Wno-linker-warnings")

if(COMPILER_SUPPORT_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
endif()

if(COMPILER_SUPPORT_COLOR_DIAGNOSTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
endif()

if(COMPILER_SUPPORT_ERROR_RETURN_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif()

if(COMPILER_SUPPORT_MARCHNATIVE)
    option(CXX_FLAG_ARCH_NATIVE "Use -march=native flag" On)
    if(CXX_FLAG_ARCH_NATIVE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
endif()


check_cxx_source_compiles("
    #include <valarray>
    int main(){}
    "
    CXX_VALARRAY_COMPILE)

# this is a check used on systems with GCC 10.2.1-6 20210110
# because of a mismatch between valarray declaration and header
# bug was created by this https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103022
# see : https://bugs.mageia.org/show_bug.cgi?id=30658
if(NOT CXX_VALARRAY_COMPILE)

    check_cxx_source_compiles("
        #include <utility>
        #include <type_traits>
        #include <algorithm>
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored \"-Wkeyword-macro\"
        #define noexcept
        #include <valarray>
        #undef noexcept
        #pragma GCC diagnostic pop
        int main(){}
        "
        CXX_VALARRAY_COMPILE_NOEXCEPT)


    if(CXX_VALARRAY_COMPILE_NOEXCEPT)
        message(STATUS "Enable noexcept fix for valarray (#define SHAMROCK_VALARRAY_FIX)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSHAMROCK_VALARRAY_FIX")
    endif()

endif()


message( " ---- Shamrock C++ config ---- ")
message( "  CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message( "  CMAKE_CXX_FLAGS_DEBUG : ${CMAKE_CXX_FLAGS_DEBUG}")
message( "  CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")
message( " ----------------------------- ")
