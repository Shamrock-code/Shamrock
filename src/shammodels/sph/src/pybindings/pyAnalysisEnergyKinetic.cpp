// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAnalysisEnergyKinetic.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::AnalysisEnergyKinetic
 */

#include "register_analysis_impl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/modules/AnalysisEnergyKinetic.hpp"
#include "sph_pybindings.hpp"
#include <memory>

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_analysisEnergyKinetic_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;
        using T     = Model<Tvec, SPHKernel>;

        py::class_<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(m, name_model.c_str())
            .def(py::init([](T &model) {
                return std::make_unique<modules::AnalysisEnergyKinetic<Tvec, SPHKernel>>(model);
            }))
            .def("get_kinetic_energy", [](modules::AnalysisEnergyKinetic<Tvec, SPHKernel> &self) {
                return self.get_kinetic_energy();
            });
    }

    template void add_analysisEnergyKinetic_instance<f64_3, shammath::M4>(
        py::module &, const std::string &);
    template void add_analysisEnergyKinetic_instance<f64_3, shammath::M6>(
        py::module &, const std::string &);
    template void add_analysisEnergyKinetic_instance<f64_3, shammath::M8>(
        py::module &, const std::string &);
    template void add_analysisEnergyKinetic_instance<f64_3, shammath::C2>(
        py::module &, const std::string &);
    template void add_analysisEnergyKinetic_instance<f64_3, shammath::C4>(
        py::module &, const std::string &);
    template void add_analysisEnergyKinetic_instance<f64_3, shammath::C6>(
        py::module &, const std::string &);

    void register_analysisEnergyKinetic_impl(py::module &msph) {
        register_analysis_impl_for_each_kernel<modules::AnalysisEnergyKinetic>(
            msph, "analysisEnergyKinetic");
    }

} // namespace shammodels::sph::pybindings
