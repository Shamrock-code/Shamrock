// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAnalysisDisc.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::modules::AnalysisDisc
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "sph_pybindings.hpp"

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_analysis_disc_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;

        using TAnalysisDisc = shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>;

        std::string disc_analysis_name = name_model + "_AnalysisDisc";
        py::class_<TAnalysisDisc>(m, disc_analysis_name.c_str())
            .def(
                "collect_data",
                [](TAnalysisDisc &self, Tscal Rmin, Tscal Rmax, u32 Nbin, ShamrockCtx &ctx) {
                    auto anal = self.compute_analysis(Rmin, Rmax, Nbin, ctx);
                    py::dict dic_out;

                    auto radius  = anal.radius.copy_to_stdvec();
                    auto counter = anal.counter.copy_to_stdvec();
                    auto Sigma   = anal.Sigma.copy_to_stdvec();
                    auto lx      = anal.lx.copy_to_stdvec();
                    auto ly      = anal.ly.copy_to_stdvec();
                    auto lz      = anal.lz.copy_to_stdvec();
                    auto tilt    = anal.tilt.copy_to_stdvec();
                    auto twist   = anal.twist.copy_to_stdvec();
                    auto psi     = anal.psi.copy_to_stdvec();
                    auto Hsq     = anal.Hsq.copy_to_stdvec();

                    dic_out["radius"]  = radius;
                    dic_out["counter"] = counter;
                    dic_out["Sigma"]   = Sigma;
                    dic_out["lx"]      = lx;
                    dic_out["ly"]      = ly;
                    dic_out["lz"]      = lz;
                    dic_out["tilt"]    = tilt;
                    dic_out["twist"]   = twist;
                    dic_out["psi"]     = psi;
                    dic_out["Hsq"]     = Hsq;

                    return dic_out;
                });
    }

    template void add_analysis_disc_instance<f64_3, shammath::M4>(
        py::module &, const std::string &);
    template void add_analysis_disc_instance<f64_3, shammath::M6>(
        py::module &, const std::string &);
    template void add_analysis_disc_instance<f64_3, shammath::M8>(
        py::module &, const std::string &);
    template void add_analysis_disc_instance<f64_3, shammath::C2>(
        py::module &, const std::string &);
    template void add_analysis_disc_instance<f64_3, shammath::C4>(
        py::module &, const std::string &);
    template void add_analysis_disc_instance<f64_3, shammath::C6>(
        py::module &, const std::string &);

} // namespace shammodels::sph::pybindings
