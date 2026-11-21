// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAnalysisSodTube.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::AnalysisSodTube
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "sph_pybindings.hpp"

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis_sod_tube_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;

        using TAnalysisSodTube = shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>;

        std::string sod_tube_analysis_name = name_model + "_AnalysisSodTube";
        py::class_<TAnalysisSodTube>(m, sod_tube_analysis_name.c_str())
            .def("compute_L2_dist", [](TAnalysisSodTube &self) -> std::tuple<Tscal, Tvec, Tscal> {
                auto ret = self.compute_L2_dist();
                return {ret.rho, ret.v, ret.P};
            });
    }

    template void add_analysis_sod_tube_instance<f64_3, shammath::M4>(
        py::module &, const std::string &);
    template void add_analysis_sod_tube_instance<f64_3, shammath::M6>(
        py::module &, const std::string &);
    template void add_analysis_sod_tube_instance<f64_3, shammath::M8>(
        py::module &, const std::string &);
    template void add_analysis_sod_tube_instance<f64_3, shammath::C2>(
        py::module &, const std::string &);
    template void add_analysis_sod_tube_instance<f64_3, shammath::C4>(
        py::module &, const std::string &);
    template void add_analysis_sod_tube_instance<f64_3, shammath::C6>(
        py::module &, const std::string &);

} // namespace shammodels::sph::pybindings
