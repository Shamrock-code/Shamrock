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
 * @file register_analysis_impl.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Helper binding an analysis factory function for every SPH kernel bound to Python.
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/Model.hpp"

namespace shammodels::sph::pybindings {

    template<class Analysis, typename Tvec, template<class> class SPHKernel>
    auto analysis_impl(shammodels::sph::Model<Tvec, SPHKernel> &model) -> Analysis {
        return Analysis(model);
    }

    template<template<class, template<class> class> class Analysis>
    void register_analysis_impl_for_each_kernel(py::module &msph, const char *name_class) {
        using namespace shammodels::sph;

        using SPHModel_f64_3_M4 = shammodels::sph::Model<f64_3, shammath::M4>;
        using SPHModel_f64_3_M6 = shammodels::sph::Model<f64_3, shammath::M6>;
        using SPHModel_f64_3_M8 = shammodels::sph::Model<f64_3, shammath::M8>;

        using SPHModel_f64_3_C2 = shammodels::sph::Model<f64_3, shammath::C2>;
        using SPHModel_f64_3_C4 = shammodels::sph::Model<f64_3, shammath::C4>;
        using SPHModel_f64_3_C6 = shammodels::sph::Model<f64_3, shammath::C6>;

        msph.def(
            name_class,
            [](SPHModel_f64_3_M4 &model) {
                return analysis_impl<Analysis<f64_3, shammath::M4>>(model);
            },
            py::kw_only(),
            py::arg("model"));

        msph.def(
            name_class,
            [](SPHModel_f64_3_M6 &model) {
                return analysis_impl<Analysis<f64_3, shammath::M6>>(model);
            },
            py::kw_only(),
            py::arg("model"));

        msph.def(
            name_class,
            [](SPHModel_f64_3_M8 &model) {
                return analysis_impl<Analysis<f64_3, shammath::M8>>(model);
            },
            py::kw_only(),
            py::arg("model"));

        msph.def(
            name_class,
            [](SPHModel_f64_3_C2 &model) {
                return analysis_impl<Analysis<f64_3, shammath::C2>>(model);
            },
            py::kw_only(),
            py::arg("model"));

        msph.def(
            name_class,
            [](SPHModel_f64_3_C4 &model) {
                return analysis_impl<Analysis<f64_3, shammath::C4>>(model);
            },
            py::kw_only(),
            py::arg("model"));

        msph.def(
            name_class,
            [](SPHModel_f64_3_C6 &model) {
                return analysis_impl<Analysis<f64_3, shammath::C6>>(model);
            },
            py::kw_only(),
            py::arg("model"));
    }

} // namespace shammodels::sph::pybindings
