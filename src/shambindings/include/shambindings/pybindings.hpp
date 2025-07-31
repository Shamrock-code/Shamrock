// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pybindings.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include <shambindings/pybindaliases.hpp>

namespace shambindings {

    /**
     * @brief Init python bindings and register them to Python API
     *
     * @param m the python module to bind definitions on
     */
    void init_lib(py::module &m);

    /**
     * @brief Init python bindings and register them to Python API
     *
     * @param m the python module to bind definitions on
     */
    void init_embed(py::module &m);

    /**
     * @brief Expect python bindings to be initialized as lib mode, throws if not
     */
    void expect_init_lib(SourceLocation loc = SourceLocation{});

    /**
     * @brief Expect python bindings to be initialized as embed mode, throws if not
     */
    void expect_init_embed(SourceLocation loc = SourceLocation{});

} // namespace shambindings
