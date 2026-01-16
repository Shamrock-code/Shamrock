## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

################################
# Shamrock version handling
################################

# Based on the versionning in AdaptiveCPP
# see : https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/CMakeLists.txt#L554

execute_process(
    COMMAND git status
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_STATUS_OUT
    RESULT_VARIABLE GIT_STATUS
)

# With a release tarball, "git status" will fail (return non zero)
if(GIT_STATUS)
    set(SHAMROCK_VERSION_SUFFIX "")
    set(SHAMROCK_FOLDER_IS_GIT 0)
else()
    set(SHAMROCK_FOLDER_IS_GIT 1)
    # Get the latest abbreviated commit hash of the working branch
    execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE SHAMROCK_GIT_COMMIT_HASH
        RESULT_VARIABLE GIT_HASH_RETURN_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # git branch
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE SHAMROCK_GIT_BRANCH
        RESULT_VARIABLE GIT_BRANCH_RETURN_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Replace slashes with underscores in branch name to avoid file path issues
    string(REPLACE "/" "_" SHAMROCK_GIT_BRANCH "${SHAMROCK_GIT_BRANCH}")
    # check whether there are local changes
    execute_process(COMMAND git diff-index --name-only HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE SHAMROCK_LOCAL_CHANGES
        RESULT_VARIABLE GIT_LOCAL_CHANGES_RETURN_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (GIT_HASH_RETURN_CODE EQUAL 0 AND GIT_BRANCH_RETURN_CODE EQUAL 0 AND
         GIT_LOCAL_CHANGES_RETURN_CODE EQUAL 0)

        if(NOT "${SHAMROCK_LOCAL_CHANGES}" STREQUAL "")
        set(DIRTY_STR ".dirty")
        else()
        set(DIRTY_STR "")
        endif()

        set(SHAMROCK_VERSION_SUFFIX "+git.${SHAMROCK_GIT_COMMIT_HASH}.${SHAMROCK_GIT_BRANCH}${DIRTY_STR}")
    endif()
endif()

set(SHAMROCK_VERSION_STRING ${SHAMROCK_VERSION_MAJOR}.${SHAMROCK_VERSION_MINOR}.${SHAMROCK_VERSION_PATCH}${SHAMROCK_VERSION_SUFFIX})
message(STATUS "Shamrock version : ${SHAMROCK_VERSION_STRING}")
