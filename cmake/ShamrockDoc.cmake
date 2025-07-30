## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------

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
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/doc/doxygen
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/doxygen/html ${CMAKE_BINARY_DIR}/doc/doxygen
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/doxygen
        DEPENDS shamrock_doc_figs
        COMMENT "Building Doxygen documentation with dox.conf and moving to build/doc/doxygen"
    )

    # Target to build mkdocs documentation (depends on doxygen)
    add_custom_target(shamrock_doc_mkdocs
        COMMAND ${CMAKE_COMMAND} -E echo "Generating MkDocs documentation..."
        COMMAND ${CMAKE_SOURCE_DIR}/doc/mkdocs/cmake_mkdocs.sh ${PYTHON_EXECUTABLE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/doc/mkdocs
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/mkdocs/site ${CMAKE_BINARY_DIR}/doc/mkdocs
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc/mkdocs
        DEPENDS shamrock_doc_doxygen
        COMMENT "Building MkDocs documentation with cmake_cmake_mkdocs.sh"
    )

    # Target to build sphinx documentation (depends on mkdocs)
    add_custom_target(shamrock_doc_sphinx
        COMMAND ${CMAKE_COMMAND} -E echo "Generating Sphinx documentation..."
        COMMAND ${CMAKE_SOURCE_DIR}/doc/sphinx/cmake_sphinx.sh ${PYTHON_EXECUTABLE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/doc/sphinx
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/doc/sphinx/build ${CMAKE_BINARY_DIR}/doc/sphinx
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS shamrock_doc_mkdocs shamrock_pylib
        COMMENT "Building Sphinx documentation with cmake_sphinx.sh"
    )

    # Target to build tmpindex (depends on sphinx)
    add_custom_target(shamrock_doc_tmpindex
        COMMAND ${CMAKE_COMMAND} -E echo "Generating doc tmpindex..."
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/doc/tmpindex.html ${CMAKE_BINARY_DIR}/doc/tmpindex.html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS shamrock_doc_sphinx
        COMMENT "Building tmpindex.html for documentation"
    )

    # Target to build all documentation (figures, doxygen, mkdocs, sphinx, tmpindex)
    add_custom_target(shamrock_doc
        DEPENDS
            shamrock_doc_tmpindex
        COMMENT "Build all Shamrock documentation (figures, Doxygen, MkDocs, Sphinx, tmpindex)"
    )

endif()
