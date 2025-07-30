## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

###############################################################################
### test git submodules
###############################################################################

option(SHAMROCK_CHECK_SUBMODULES_COMMIT_HASH "Check the commit hash of the git submodules" On)

function(_check_git_submodule_cloned directory expect_hash)
    file(GLOB files RELATIVE "${directory}" "${directory}/*")
    if(NOT files)

        message(FATAL_ERROR
            "The git submodule '${directory}' is empty\n"
            "please do : git submodule update --init --recursive\n"
        )

    endif()

    if(SHAMROCK_FOLDER_IS_GIT AND SHAMROCK_CHECK_SUBMODULES_COMMIT_HASH)
        execute_process(
            COMMAND git log -1 --format=%h
            WORKING_DIRECTORY "${directory}"
            OUTPUT_VARIABLE submodule_commit_hash
            RESULT_VARIABLE GIT_HASH_RETURN_CODE
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        #message(STATUS "Submodule '${directory}' commit hash: ${submodule_commit_hash}")

        if(NOT "${submodule_commit_hash}" STREQUAL "${expect_hash}")
            message(FATAL_ERROR
                "The git submodule '${directory}' is not in sync\n"
                "current commit hash '${submodule_commit_hash}' is not equal to expected '${expect_hash}'\n"
                "please do : git pull --recurse-submodules\n"
            )
        endif()
    endif()

    #message(STATUS "The subdirectory '${directory}' contains ${files}")
endfunction()
