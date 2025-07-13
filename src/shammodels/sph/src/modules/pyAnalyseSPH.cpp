// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAnalyseSPH.cpp
 * @author David Fang (fang.david03@gmail.com)
 * @brief Create python method shamrock.analyseSPH(model) which returns an AnalysisSPH instance
 *
 */

#include "shambindings/pybindaliases.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/modules/AnalysisSPH.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using SPHModel_f64_3_M4 = shammodels::sph::Model<f64_3, shammath::M4>;
using SPHModel_f64_3_M6 = shammodels::sph::Model<f64_3, shammath::M6>;
using SPHModel_f64_3_M8 = shammodels::sph::Model<f64_3, shammath::M8>;

using VariantAnalysisSPH = std::variant<
    shammodels::sph::modules::AnalysisSPH<f64_3, shammath::M4>,
    shammodels::sph::modules::AnalysisSPH<f64_3, shammath::M6>,
    shammodels::sph::modules::AnalysisSPH<f64_3, shammath::M8>>;

template<class Tvec, template<class> class SPHKernel>
void add_analysisSPH_instance(py::module &m, std::string name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T = Model<Tvec, SPHKernel>;

    py::class_<modules::AnalysisSPH<Tvec, SPHKernel>>(m, name_model.c_str())
        .def(py::init([](T &model) {
            return std::make_unique<modules::AnalysisSPH<Tvec, SPHKernel>>(model);
        }))
        .def("get_barycenter", [](modules::AnalysisSPH<Tvec, SPHKernel> &self) {
            auto result = self.get_barycenter();
            auto dim    = self.dim;
            py::array_t<Tscal> numpy_barycenter({dim});
            for (u32 i = 0; i < dim; i++) {
                numpy_barycenter.mutable_at(i) = result.barycenter[i];
            }
            return py::make_tuple(numpy_barycenter, result.mass_disc);
        });
}

using namespace shammodels::sph;

template<typename Tvec, template<class> class SPHKernel>
auto analyseSPH_impl(shammodels::sph::Model<Tvec, SPHKernel> &model)
    -> modules::AnalysisSPH<Tvec, SPHKernel> {
    return modules::AnalysisSPH<Tvec, SPHKernel>(model);
}

Register_pymod(pyanalysisSPH) {
    shamlog_debug_ln("[Py]", "registering shamrock.analyseSPH()");

    add_analysisSPH_instance<f64_3, shammath::M4>(m, "AnalysisSPH_f64_3_M4");
    add_analysisSPH_instance<f64_3, shammath::M6>(m, "AnalysisSPH_f64_3_M6");
    add_analysisSPH_instance<f64_3, shammath::M8>(m, "AnalysisSPH_f64_3_M8");

    using namespace shammodels::sph;

    m.def(
        "analyseSPH",
        [](SPHModel_f64_3_M4 &model) {
            return analyseSPH_impl<f64_3, shammath::M4>(model);
        },
        py::kw_only(),
        py::arg("model"));

    m.def(
        "analyseSPH",
        [](SPHModel_f64_3_M6 &model) {
            return analyseSPH_impl<f64_3, shammath::M6>(model);
        },
        py::kw_only(),
        py::arg("model"));

    m.def(
        "analyseSPH",
        [](SPHModel_f64_3_M8 &model) {
            return analyseSPH_impl<f64_3, shammath::M8>(model);
        },
        py::kw_only(),
        py::arg("model"));
}
