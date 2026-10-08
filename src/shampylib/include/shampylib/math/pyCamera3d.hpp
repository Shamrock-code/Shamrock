// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pyCamera3d.hpp
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"

namespace shampylib {

    template<class Tvec>
    void init_shamrock_math_Camera3d(py::module &m, std::string name);

} // namespace shampylib
