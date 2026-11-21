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
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Private declarations of the per-class registration functions of the `model_sph`
 * Python bindings.
 *
 * Each function is defined in its own translation unit (`pybindings/py<Class>.cpp`), with
 * explicit instantiations for the SPH kernels bound to Python, so that the pybind11 code of each
 * class is compiled in parallel. They are all called, in a fixed order, from the single
 * `ON_PYTHON_INIT` in `pySPHModel.cpp`: pybind11 builds signature docstrings from the types
 * registered at definition time, so the registration order must not change.
 */

// pybind11 type casters must be visible in every TU that binds code, otherwise conversions of
// std::function, std::optional, numpy arrays, ... silently fail at import time (or worse, differ
// between TUs). This is the exact set of casters the bindings were compiled with when they all
// lived in a single pySPHModel.cpp, so keep it identical in every per-class TU.
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

namespace shammodels::sph::pybindings {

    // Kernel-templated classes, defined & explicitly instantiated in pybindings/py<Class>.cpp

    template<class Tvec, template<class> class SPHKernel>
    void log_instance_registration(const std::string &name_config, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_solver_config_instance(py::module &m, const std::string &name_config);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis_sod_tube_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis_disc_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_sph_setup_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_model_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisBarycenter_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisEnergyKinetic_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisEnergyPotential_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisTotalMomentum_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisAngularMomentum_instance(py::module &m, const std::string &name_model);

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisDustMass_instance(py::module &m, const std::string &name_model);

    // Analysis factory functions (one overload per kernel), defined in pybindings/pyAnalysis*.cpp

    void register_analysisBarycenter_impl(py::module &msph);
    void register_analysisEnergyKinetic_impl(py::module &msph);
    void register_analysisEnergyPotential_impl(py::module &msph);
    void register_analysisTotalMomentum_impl(py::module &msph);
    void register_analysisAngularMomentum_impl(py::module &msph);
    void register_analysisDustMass_impl(py::module &msph);

    // Non-templated classes & functions

    void add_patch_data_lazy_getter(py::module &m);
    void add_evolve_until_results(py::module &m);
    void add_get_model_sph(py::module &m);
    void add_isph_setup_node(py::module &msph);
    void add_timestep_log(py::module &msph);

} // namespace shammodels::sph::pybindings
