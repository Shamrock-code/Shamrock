// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "pybindings/sph_pybindings.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shammath/sphkernels.hpp"
#include <string>

using namespace shammodels::sph::pybindings;

template<class Tvec, template<class> class SPHKernel>
void add_instance(py::module &m, std::string name_config, std::string name_model) {
    log_instance_registration<Tvec, SPHKernel>(name_config, name_model);

    add_solver_config_instance<Tvec, SPHKernel>(m, name_config);
    add_analysis_sod_tube_instance<Tvec, SPHKernel>(m, name_model);
    add_analysis_disc_instance<Tvec, SPHKernel>(m, name_model);
    add_sph_setup_instance<Tvec, SPHKernel>(m, name_model);
    add_model_instance<Tvec, SPHKernel>(m, name_model);
}

// Registration order matters (pybind11 docstrings are built from the types registered at
// definition time), keep this single ON_PYTHON_INIT and the call order below unchanged.
ON_PYTHON_INIT {
    auto &m = root_module;

    py::module msph = m.def_submodule("model_sph", "Shamrock sph solver");

    add_patch_data_lazy_getter(m);

    add_evolve_until_results(m);

    add_instance<f64_3, shammath::M4>(msph, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    add_instance<f64_3, shammath::M6>(msph, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");
    add_instance<f64_3, shammath::M8>(msph, "SPHModel_f64_3_M8_SolverConfig", "SPHModel_f64_3_M8");

    add_instance<f64_3, shammath::C2>(msph, "SPHModel_f64_3_C2_SolverConfig", "SPHModel_f64_3_C2");
    add_instance<f64_3, shammath::C4>(msph, "SPHModel_f64_3_C4_SolverConfig", "SPHModel_f64_3_C4");
    add_instance<f64_3, shammath::C6>(msph, "SPHModel_f64_3_C6_SolverConfig", "SPHModel_f64_3_C6");

    add_get_model_sph(m);

    add_isph_setup_node(msph);

    add_timestep_log(msph);

    add_analysisBarycenter_instance<f64_3, shammath::M4>(msph, "AnalysisBarycenter_f64_3_M4");
    add_analysisBarycenter_instance<f64_3, shammath::M6>(msph, "AnalysisBarycenter_f64_3_M6");
    add_analysisBarycenter_instance<f64_3, shammath::M8>(msph, "AnalysisBarycenter_f64_3_M8");

    add_analysisBarycenter_instance<f64_3, shammath::C2>(msph, "AnalysisBarycenter_f64_3_C2");
    add_analysisBarycenter_instance<f64_3, shammath::C4>(msph, "AnalysisBarycenter_f64_3_C4");
    add_analysisBarycenter_instance<f64_3, shammath::C6>(msph, "AnalysisBarycenter_f64_3_C6");

    add_analysisEnergyKinetic_instance<f64_3, shammath::M4>(msph, "AnalysisEnergyKinetic_f64_3_M4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M6>(msph, "AnalysisEnergyKinetic_f64_3_M6");
    add_analysisEnergyKinetic_instance<f64_3, shammath::M8>(msph, "AnalysisEnergyKinetic_f64_3_M8");

    add_analysisEnergyKinetic_instance<f64_3, shammath::C2>(msph, "AnalysisEnergyKinetic_f64_3_C2");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C4>(msph, "AnalysisEnergyKinetic_f64_3_C4");
    add_analysisEnergyKinetic_instance<f64_3, shammath::C6>(msph, "AnalysisEnergyKinetic_f64_3_C6");

    add_analysisEnergyPotential_instance<f64_3, shammath::M4>(
        msph, "AnalysisEnergyPotential_f64_3_M4");
    add_analysisEnergyPotential_instance<f64_3, shammath::M6>(
        msph, "AnalysisEnergyPotential_f64_3_M6");
    add_analysisEnergyPotential_instance<f64_3, shammath::M8>(
        msph, "AnalysisEnergyPotential_f64_3_M8");

    add_analysisEnergyPotential_instance<f64_3, shammath::C2>(
        msph, "AnalysisEnergyPotential_f64_3_C2");
    add_analysisEnergyPotential_instance<f64_3, shammath::C4>(
        msph, "AnalysisEnergyPotential_f64_3_C4");
    add_analysisEnergyPotential_instance<f64_3, shammath::C6>(
        msph, "AnalysisEnergyPotential_f64_3_C6");

    add_analysisTotalMomentum_instance<f64_3, shammath::M4>(msph, "AnalysisTotalMomentum_f64_3_M4");
    add_analysisTotalMomentum_instance<f64_3, shammath::M6>(msph, "AnalysisTotalMomentum_f64_3_M6");
    add_analysisTotalMomentum_instance<f64_3, shammath::M8>(msph, "AnalysisTotalMomentum_f64_3_M8");

    add_analysisTotalMomentum_instance<f64_3, shammath::C2>(msph, "AnalysisTotalMomentum_f64_3_C2");
    add_analysisTotalMomentum_instance<f64_3, shammath::C4>(msph, "AnalysisTotalMomentum_f64_3_C4");
    add_analysisTotalMomentum_instance<f64_3, shammath::C6>(msph, "AnalysisTotalMomentum_f64_3_C6");

    add_analysisAngularMomentum_instance<f64_3, shammath::M4>(
        msph, "AnalysisAngularMomentum_f64_3_M4");
    add_analysisAngularMomentum_instance<f64_3, shammath::M6>(
        msph, "AnalysisAngularMomentum_f64_3_M6");
    add_analysisAngularMomentum_instance<f64_3, shammath::M8>(
        msph, "AnalysisAngularMomentum_f64_3_M8");

    add_analysisAngularMomentum_instance<f64_3, shammath::C2>(
        msph, "AnalysisAngularMomentum_f64_3_C2");
    add_analysisAngularMomentum_instance<f64_3, shammath::C4>(
        msph, "AnalysisAngularMomentum_f64_3_C4");
    add_analysisAngularMomentum_instance<f64_3, shammath::C6>(
        msph, "AnalysisAngularMomentum_f64_3_C6");

    register_analysisBarycenter_impl(msph);
    register_analysisEnergyKinetic_impl(msph);
    register_analysisEnergyPotential_impl(msph);
    register_analysisTotalMomentum_impl(msph);
    register_analysisAngularMomentum_impl(msph);

    add_analysisDustMass_instance<f64_3, shammath::M4>(msph, "AnalysisDustMass_f64_3_M4");
    add_analysisDustMass_instance<f64_3, shammath::M6>(msph, "AnalysisDustMass_f64_3_M6");
    add_analysisDustMass_instance<f64_3, shammath::M8>(msph, "AnalysisDustMass_f64_3_M8");

    add_analysisDustMass_instance<f64_3, shammath::C2>(msph, "AnalysisDustMass_f64_3_C2");
    add_analysisDustMass_instance<f64_3, shammath::C4>(msph, "AnalysisDustMass_f64_3_C4");
    add_analysisDustMass_instance<f64_3, shammath::C6>(msph, "AnalysisDustMass_f64_3_C6");

    register_analysisDustMass_impl(msph);
}
