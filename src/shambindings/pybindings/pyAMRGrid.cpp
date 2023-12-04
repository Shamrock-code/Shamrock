// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRGrid.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shamrock/amr/AMRGrid.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

#include <pybind11/numpy.h>
#include "shambindings/pybind11_stl.hpp"

Register_pymod(pyamrgridinit) {

    using Grid = shamrock::amr::AMRGrid<u64_3, 3>;

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
