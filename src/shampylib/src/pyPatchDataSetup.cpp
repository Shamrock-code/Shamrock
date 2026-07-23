// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyPatchDataSetup.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Python bindings for PatchDataSetup
 */

#include "shambindings/pybindaliases.hpp"
#include "shamcomm/logs.hpp"
#include "shampylib/PatchDataSetup.hpp"

ON_PYTHON_INIT {
    auto &m = root_module;

    shamlog_debug_ln("[Py]", "registering shamrock.PatchDataSetup");

    py::class_<shamrock::PatchDataSetup>(m, "PatchDataSetup")
        .def("get", &shamrock::PatchDataSetup::get, py::arg("name"))
        .def("set", &shamrock::PatchDataSetup::set, py::arg("name"), py::arg("value"));
}
