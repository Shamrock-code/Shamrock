## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- Pybind11 section ----")

######################
# Python selection part
######################

if(NOT DEFINED PYTHON_EXECUTABLE)
    message(WARNING "PYTHON_EXECUTABLE is not defined, please set it as follows:\n"
        "-DPYTHON_EXECUTABLE=$(python3 -c \"import sys; print(sys.executable)\")")
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Trying to autodetect python executable: python3")
    execute_process(COMMAND python3 -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE PYTHON_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE STATUS
    )
    if(STATUS AND NOT STATUS EQUAL 0)
        message(STATUS "Failed autodetect with python3:\n ${STATUS}")
        unset(PYTHON_EXECUTABLE)
    endif()
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
    message(STATUS "Trying to autodetect python executable: python")
    execute_process(COMMAND python -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE PYTHON_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE STATUS
    )
    if(STATUS AND NOT STATUS EQUAL 0)
        message(STATUS "Failed autodetect with python:\n ${STATUS}")
        unset(PYTHON_EXECUTABLE)
    endif()
endif()

message(STATUS "PYTHON_EXECUTABLE : ${PYTHON_EXECUTABLE}")

if(NOT DEFINED PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE is not defined and autodetect failed, please set it as follows:\n"
        "-DPYTHON_EXECUTABLE=$(python3 -c \"import sys; print(sys.executable)\")")
endif()

# Cache the result to avoid rerunning detection next time
set(PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE INTERNAL "")

###############################################################################
### pybind11
###############################################################################

option(SHAMROCK_EXTERNAL_PYBIND11 "use pybind11 lib from the host system" Off)

if(SHAMROCK_EXTERNAL_PYBIND11)
    message(STATUS "Using system pybind11")
    find_package(pybind11 REQUIRED)
else()
    message(STATUS "Using git submodule pybind11")

    _check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11 a2e59f0e7065404b44dfe92a28aca47ba1378dc4)

    add_subdirectory(external/pybind11)
endif()
