// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShamcomm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamalgs/collective/gather_str.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/wrapper.hpp"
#include <pybind11/pytypes.h>
#include <unordered_map>
#include <utility>
#include <vector>

ON_PYTHON_INIT {

    py::module shamcomm_module = root_module.def_submodule("comm", "comm library");

    shamcomm_module.def("get_timer", [](std::string name) {
        return shamcomm::mpi::get_timer(std::move(name));
    });

    shamcomm_module.def("get_timers", []() {
        return shamcomm::mpi::get_timers();
    });

    shamcomm_module.def(
        "mpi_timers_delta",
        [](std::unordered_map<std::string, f64> start, std::unordered_map<std::string, f64> end) {
            std::unordered_map<std::string, f64> deltas{};

            for (auto &k : shamcomm::mpi::get_possible_keys()) {
                deltas[k] = shamalgs::collective::allreduce_max(end[k] - start[k]);
            }

            return deltas;
        });
}
