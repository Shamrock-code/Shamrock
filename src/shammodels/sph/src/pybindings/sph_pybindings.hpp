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
 * @file sph_pybindings.hpp
 * @author David Fang (david.fang@ikmail.com) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Private declarations of the `model_sph` bindings split out of `pySPHModel.cpp`.
 *
 * The SolverConfig and Model bindings are by far the heaviest to compile, so they live in their
 * own translation units (`pybindings/pySolverConfig.cpp` and `pybindings/pyModel.cpp`), with
 * explicit instantiations for the SPH kernels bound to Python, and build in parallel with
 * `pySPHModel.cpp`. They are called from `add_instance` in `pySPHModel.cpp`, under its single
 * `ON_PYTHON_INIT`: pybind11 builds signature docstrings from the types registered at definition
 * time, so the registration order must not change.
 */

// pybind11 type casters must be visible in every TU that binds code, otherwise conversions of
// std::function, std::optional, numpy arrays, ... silently fail at import time (or worse, differ
// between TUs). This is the exact set of casters the bindings were compiled with when they all
// lived in a single pySPHModel.cpp, so keep it identical in every TU including this header.
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace shammodels::sph::pybindings {

    /// Bind `SolverConfig<Tvec, SPHKernel>` (defined in pybindings/pySolverConfig.cpp)
    template<class Tvec, template<class> class SPHKernel>
    void add_solver_config_instance(py::module &m, const std::string &name_config);

    /// Bind `Model<Tvec, SPHKernel>` (defined in pybindings/pyModel.cpp)
    template<class Tvec, template<class> class SPHKernel>
    void add_model_instance(py::module &m, const std::string &name_model);

} // namespace shammodels::sph::pybindings
