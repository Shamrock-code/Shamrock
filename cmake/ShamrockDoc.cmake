## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

message("${as_subproject}")

find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)

    # Target to generate figures for documentation
    add_custom_target(shamrock_doc_figs
        COMMAND ${CMAKE_COMMAND} -E echo "Generating figures for documentation..."
        COMMAND bash ${CMAKE_SOURCE_DIR}/doc/mkdocs/docs/assets/figures/make_all_figs.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/mkdocs/docs/assets/figures
        COMMENT "Building documentation figures with make_all_figs.sh"
    )

    # Target to build doxygen documentation (depends on figures)
    add_custom_target(shamrock_doc_doxygen
        COMMAND ${CMAKE_COMMAND} -E echo "Generating Doxygen documentation..."
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/doc/doxygen/dox.conf
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/doxygen
        DEPENDS shamrock_doc_figs
        COMMENT "Building Doxygen documentation with dox.conf"
    )

endif()
