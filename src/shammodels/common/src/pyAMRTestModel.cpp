// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRTestModel.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shammodels/common/amr/AMROverheadtest.hpp"
#include <pybind11/numpy.h>
#include <memory>

Register_pymod(pyamrtestmode) {

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.AMRTestModel");
    py::class_<AMRTestModel>(m, "AMRTestModel")
        .def(py::init([](AMRTestModel::Grid &grd) {
            return std::make_unique<AMRTestModel>(grd);
        }))
        .def(
            "refine",
            [](AMRTestModel &obj) {
                obj.refine();
            })
        .def(
            "derefine",
            [](AMRTestModel &obj) {
                obj.derefine();
            })
        .def("step", [](AMRTestModel &obj) {
            obj.step();
        });
}
