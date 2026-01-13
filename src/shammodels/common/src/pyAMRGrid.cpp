// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRGrid.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <pybind11/numpy.h>

Register_pymod(pyamrgridinit) {

    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.AMRGrid");
    py::class_<Grid>(m, "AMRGrid")
        .def(py::init([](ShamrockCtx &ctx) {
            return std::make_unique<Grid>(*ctx.sched);
        }))
        .def(
            "make_base_grid",
            [](Grid &grid,
               std::array<u64, 3> min,
               std::array<u64, 3> cell_size,
               std::array<u32, 3> cell_count) {
                grid.make_base_grid(
                    u64_3{min[0], min[1], min[2]},
                    u64_3{cell_size[0], cell_size[1], cell_size[2]},
                    cell_count);
            });
}
