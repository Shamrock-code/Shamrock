// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRZeusModel.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/zeus/Model.hpp"
#include "shammodels/zeus/modules/AnalysisSodTube.hpp"
#include <pybind11/functional.h>
#include <memory>

namespace shammodels::zeus {
    template<class Tvec, class TgridVec>
    void add_instance(py::module &m, std::string name_config, std::string name_model) {

        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        using T                = Model<Tvec, TgridVec>;
        using TConfig          = typename T::Solver::Config;
        using TAnalysisSodTube = shammodels::zeus::modules::AnalysisSodTube<Tvec, TgridVec>;

        shamlog_debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
        shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

        py::class_<TConfig>(m, name_config.c_str())
            .def(
                "set_scale_factor",
                [](TConfig &self, Tscal scale_factor) {
                    self.grid_coord_to_pos_fact = scale_factor;
                })
            .def(
                "set_eos_gamma",
                [](TConfig &self, Tscal eos_gamma) {
                    self.set_eos_gamma(eos_gamma);
                })
            .def(
                "set_consistent_transport",
                [](TConfig &self, bool enable) {
                    self.use_consistent_transport = enable;
                })
            .def("set_van_leer", [](TConfig &self, bool enable) {
                self.use_van_leer = enable;
            });

        std::string sod_tube_analysis_name = name_model + "_AnalysisSodTube";
        py::class_<TAnalysisSodTube>(m, sod_tube_analysis_name.c_str())
            .def("compute_L2_dist", [](TAnalysisSodTube &self) -> std::tuple<Tscal, Tvec, Tscal> {
                auto ret = self.compute_L2_dist();
                return {ret.rho, ret.v, ret.P};
            });
        py::class_<T>(m, name_model.c_str())
            .def("init_scheduler", &T::init_scheduler)
            .def("make_base_grid", &T::make_base_grid)
            .def("dump_vtk", &T::dump_vtk)
            .def("evolve_once", &T::evolve_once)
            .def("set_field_value_lambda_f64", &T::template set_field_value_lambda<f64>)
            .def("set_field_value_lambda_f64_3", &T::template set_field_value_lambda<f64_3>)
            .def(
                "gen_default_config",
                [](T &self) -> TConfig {
                    return TConfig();
                })
            .def(
                "set_solver_config",
                [](T &self, TConfig cfg) {
                    if (self.ctx.is_scheduler_initialized()) {
                        shambase::throw_with_loc<std::runtime_error>(
                            "Cannot change solver config after scheduler is initialized");
                    }
                    cfg.check_config();
                    self.solver.solver_config = cfg;
                })
            .def(
                "get_cell_coords",
                [](T &self, std::pair<TgridVec, TgridVec> block_coord, u32 cell_local_id) {
                    return self.get_cell_coords(block_coord, cell_local_id);
                })
            .def(
                "make_analysis_sodtube",
                [](T &self,
                   shamphys::SodTube sod,
                   Tvec direction,
                   Tscal time_val,
                   Tscal x_ref,
                   Tscal x_min,
                   Tscal x_max) {
                    return std::make_unique<TAnalysisSodTube>(
                        self.ctx,
                        self.solver.solver_config,
                        self.solver.storage,
                        sod,
                        direction,
                        time_val,
                        x_ref,
                        x_min,
                        x_max);
                });
    }
} // namespace shammodels::zeus

Register_pymod(pyamrzeusmodel) {

    py::module mzeus = m.def_submodule("model_zeus", "Shamrock Zeus solver");

    std::string base_name = "ZeusModel";
    using namespace shammodels::zeus;

    add_instance<f64_3, i64_3>(
        mzeus, base_name + "_f64_3_i64_3_SolverConfig", base_name + "_f64_3_i64_3_Model");

    using VariantAMRZeusBind = std::variant<std::unique_ptr<Model<f64_3, i64_3>>>;

    m.def(
        "get_Model_Zeus",
        [](ShamrockCtx &ctx, std::string vector_type, std::string grid_repr) -> VariantAMRZeusBind {
            VariantAMRZeusBind ret;

            if (vector_type == "f64_3" && grid_repr == "i64_3") {
                ret = std::make_unique<Model<f64_3, i64_3>>(ctx);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "unknown combination of representation and grid_repr");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("grid_repr"));
}
