// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyTimestepLog.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::TimestepLog
 */

#include "shambindings/pybindaliases.hpp"
#include "shammodels/sph/Solver.hpp"
#include "sph_pybindings.hpp"

namespace shammodels::sph::pybindings {

    void add_timestep_log(py::module &msph) {
        py::class_<shammodels::sph::TimestepLog>(msph, "TimestepLog")
            .def(py::init<>())
            .def_readwrite("rank", &shammodels::sph::TimestepLog::rank)
            .def_readwrite("rate", &shammodels::sph::TimestepLog::rate)
            .def_readwrite("npart", &shammodels::sph::TimestepLog::npart)
            .def_readwrite("tcompute", &shammodels::sph::TimestepLog::tcompute)
            .def("rate_sum", &shammodels::sph::TimestepLog::rate_sum)
            .def("npart_sum", &shammodels::sph::TimestepLog::npart_sum);
    }

} // namespace shammodels::sph::pybindings
