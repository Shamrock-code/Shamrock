// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAnalysisEnergyPotential.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::AnalysisEnergyPotential
 */

#include "register_analysis_impl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/modules/AnalysisEnergyPotential.hpp"
#include "sph_pybindings.hpp"
#include <memory>

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisEnergyPotential_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;
        using T     = Model<Tvec, SPHKernel>;

        py::class_<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(m, name_model.c_str())
            .def(py::init([](T &model) {
                return std::make_unique<modules::AnalysisEnergyPotential<Tvec, SPHKernel>>(model);
            }))
            .def(
                "get_potential_energy",
                [](modules::AnalysisEnergyPotential<Tvec, SPHKernel> &self) {
                    return self.get_potential_energy();
                });
    }

    template void add_analysisEnergyPotential_instance<f64_3, shammath::M4>(
        py::module &, const std::string &);
    template void add_analysisEnergyPotential_instance<f64_3, shammath::M6>(
        py::module &, const std::string &);
    template void add_analysisEnergyPotential_instance<f64_3, shammath::M8>(
        py::module &, const std::string &);
    template void add_analysisEnergyPotential_instance<f64_3, shammath::C2>(
        py::module &, const std::string &);
    template void add_analysisEnergyPotential_instance<f64_3, shammath::C4>(
        py::module &, const std::string &);
    template void add_analysisEnergyPotential_instance<f64_3, shammath::C6>(
        py::module &, const std::string &);

    void register_analysisEnergyPotential_impl(py::module &msph) {
        register_analysis_impl_for_each_kernel<modules::AnalysisEnergyPotential>(
            msph, "analysisEnergyPotential");
    }

} // namespace shammodels::sph::pybindings
