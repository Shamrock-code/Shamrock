## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("   ---- nlohmann_json section ----")

###############################################################################
### nlohmann_json
###############################################################################

option(SHAMROCK_EXTERNAL_JSON "use nlohmann_json lib from the host system" Off)

message(STATUS "SHAMROCK_EXTERNAL_JSON : ${SHAMROCK_EXTERNAL_JSON}")

if(SHAMROCK_EXTERNAL_JSON)
    find_package(nlohmann_json 3.11.3 REQUIRED)
else()
    set(JSON_BuildTests OFF CACHE INTERNAL "")

    _check_git_submodule_cloned(${CMAKE_CURRENT_SOURCE_DIR}/external/nlohmann_json 9cca280a)

    add_subdirectory(external/nlohmann_json)
endif()
