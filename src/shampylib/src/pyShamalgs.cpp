// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamalgs.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_float.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamalgs/random.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include <pybind11/complex.h>

Register_pymod(shamalgslibinit) {

    py::module shamalgs_module = m.def_submodule("algs", "algorithmic library");

    py::class_<std::mt19937>(shamalgs_module, "rng");

    shamalgs_module.def("gen_seed", [](u64 seed) {
        return std::mt19937(seed);
    });

    shamalgs_module.def("mock_gaussian", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian<f64>(eng);
    });
    shamalgs_module.def("mock_gaussian_f64_2", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian_multidim<f64_2>(eng);
    });
    shamalgs_module.def("mock_gaussian_f64_3", [](std::mt19937 &eng) {
        return shamalgs::random::mock_gaussian_multidim<f64_3>(eng);
    });
    shamalgs_module.def("mock_unit_vector_f64_3", [](std::mt19937 &eng) {
        return shamalgs::random::mock_unit_vector<f64_3>(eng);
    });
}
