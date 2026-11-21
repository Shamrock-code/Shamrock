// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyEvolveUntilResults.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::EvolveUntilResults
 */

#include "sham/format/format.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shammodels/sph/Solver.hpp"
#include "sph_pybindings.hpp"

namespace shammodels::sph::pybindings {

    void add_evolve_until_results(py::module &m) {
        py::class_<EvolveUntilResults>(m, "EvolveUntilResults")
            .def_readwrite("reach_target_time", &EvolveUntilResults::reach_target_time)
            .def_readwrite("reach_niter_max", &EvolveUntilResults::reach_niter_max)
            .def_readwrite("reach_max_walltime", &EvolveUntilResults::reach_max_walltime)
            .def_readwrite("iter_count", &EvolveUntilResults::iter_count)
            .def("__repr__", [](const EvolveUntilResults &self) {
                return sham::format(
                    "EvolveUntilResults(reach_target_time={}, reach_niter_max={}, "
                    "reach_max_walltime={}, iter_count={})",
                    self.reach_target_time,
                    self.reach_niter_max,
                    self.reach_max_walltime,
                    self.iter_count);
            });
    }

} // namespace shammodels::sph::pybindings
