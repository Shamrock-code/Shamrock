// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyRamsesModel.cpp
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/ramses/Model.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/AnalysisSodTube.hpp"
#include <pybind11/functional.h>
#include <memory>

namespace shammodels::basegodunov {
    template<class Tvec, class TgridVec>
    void add_instance(py::module &m, std::string name_config, std::string name_model) {

        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        using T                = Model<Tvec, TgridVec>;
        using TConfig          = typename T::Solver::Config;
        using TAnalysisSodTube = shammodels::basegodunov::modules::AnalysisSodTube<Tvec, TgridVec>;

        shamlog_debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
        shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

        py::class_<TConfig>(m, name_config.c_str())
            .def(
                "set_scale_factor",
                [](TConfig &self, Tscal scale_factor) {
                    self.grid_coord_to_pos_fact = scale_factor;
                })
            .def(
                "set_Csafe",
                [](TConfig &self, Tscal Csafe) {
                    self.Csafe = Csafe;
                })
            .def(
                "set_eos_gamma",
                [](TConfig &self, Tscal eos_gamma) {
                    self.set_eos_gamma(eos_gamma);
                })
            .def(
                "set_riemann_solver_hll",
                [](TConfig &self) {
                    self.riemman_config = HLL;
                })
            .def(
                "set_riemann_solver_hllc",
                [](TConfig &self) {
                    self.riemman_config = HLLC;
                })
            .def(
                "set_riemann_solver_rusanov",
                [](TConfig &self) {
                    self.riemman_config = Rusanov;
                })
            .def(
                "set_slope_lim_none",
                [](TConfig &self) {
                    self.slope_config = None;
                })
            .def(
                "set_slope_lim_vanleer_f",
                [](TConfig &self) {
                    self.slope_config = VanLeer_f;
                })
            .def(
                "set_slope_lim_vanleer_std",
                [](TConfig &self) {
                    self.slope_config = VanLeer_std;
                })
            .def(
                "set_slope_lim_vanleer_sym",
                [](TConfig &self) {
                    self.slope_config = VanLeer_sym;
                })
            .def(
                "set_slope_lim_minmod",
                [](TConfig &self) {
                    self.slope_config = Minmod;
                })
            .def(
                "set_face_time_interpolation",
                [](TConfig &self, bool face_time_interpolate) {
                    self.face_half_time_interpolation = face_time_interpolate;
                })

            .def(
                "set_dust_mode_dhll",
                [](TConfig &self, u32 ndust) {
                    self.dust_config = {DHLL, ndust};
                })
            .def(
                "set_dust_mode_hb",
                [](TConfig &self, u32 ndust) {
                    self.dust_config = {HB, ndust};
                })
            .def(
                "set_dust_mode_none",
                [](TConfig &self) {
                    self.dust_config = {NoDust, 0};
                })
            .def(
                "set_alpha_values",
                [](TConfig &self, f32 alpha_values) {
                    return self.set_alphas_static(alpha_values);
                })
            .def(
                "set_drag_mode_no_drag",
                [](TConfig &self) {
                    self.drag_config.drag_solver_config        = NoDrag;
                    self.drag_config.enable_frictional_heating = false;
                })
            .def(
                "set_drag_mode_irk1",
                [](TConfig &self, bool frictional_status) {
                    self.drag_config.drag_solver_config        = IRK1;
                    self.drag_config.enable_frictional_heating = frictional_status;
                })
            .def(
                "set_drag_mode_irk2",
                [](TConfig &self, bool frictional_status) {
                    self.drag_config.drag_solver_config        = IRK2;
                    self.drag_config.enable_frictional_heating = frictional_status;
                })
            .def(
                "set_drag_mode_expo",
                [](TConfig &self, bool frictional_status) {
                    self.drag_config.drag_solver_config        = EXPO;
                    self.drag_config.enable_frictional_heating = frictional_status;
                })
            .def(
                "set_amr_mode_none",
                [](TConfig &self) {
                    self.amr_mode.set_refine_none();
                })
            .def(
                "set_amr_mode_density_based",
                [](TConfig &self, Tscal crit_mass) {
                    self.amr_mode.set_refine_density_based(crit_mass);
                },
                py::kw_only(),
                py::arg("crit_mass"))
            .def(
                "set_gravity_mode_no_gravity",
                [](TConfig &self) {
                    self.gravity_config.gravity_mode = NoGravity;
                })
            .def(
                "set_gravity_mode_cg",
                [](TConfig &self) {
                    self.gravity_config.gravity_mode = CG;
                })
            .def(
                "set_gravity_mode_pcg",
                [](TConfig &self) {
                    self.gravity_config.gravity_mode = PCG;
                })
            .def(
                "set_gravity_mode_bicgstab",
                [](TConfig &self) {
                    self.gravity_config.gravity_mode = BICGSTAB;
                })
            .def(
                "set_npscal_gas",
                [](TConfig &self, u32 npscal_gas) {
                    self.npscal_gas_config.npscal_gas = npscal_gas;
                })
            .def(
                "set_self_gravity_G_values",
                [](TConfig &self, bool set_G, f32 G_values) {
                    self.gravity_config.set_G = set_G;
                    self.gravity_config.G     = G_values;
                })
            .def(
                "set_self_gravity_Niter_max",
                [](TConfig &self, u32 Niter) {
                    self.gravity_config.Niter_max = Niter;
                })
            .def(
                "set_self_gravity_tol",
                [](TConfig &self, f32 tol) {
                    self.gravity_config.tol = tol;
                })
            .def("set_self_gravity_happy_breakdown_tol", [](TConfig &self, f32 tol) {
                self.gravity_config.tol_hp_bk = tol;
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
            .def("dump", &T::dump)
            .def("load_from_dump", &T::load_from_dump)
            .def("evolve_once_override_time", &T::evolve_once_time_expl)
            .def("evolve_once", &T::evolve_once)
            .def(
                "evolve_until",
                [](T &self, f64 target_time, i32 niter_max) {
                    return self.evolve_until(target_time, niter_max);
                },
                py::arg("target_time"),
                py::kw_only(),
                py::arg("niter_max") = -1)
            .def("timestep", &T::timestep)
            // .def("set_field_value_lambda_f64", &T::template set_field_value_lambda<f64>)
            // .def("set_field_value_lambda_f64_3", &T::template set_field_value_lambda<f64_3>)
            .def(
                "set_field_value_lambda_f64",
                [](T &self,
                   std::string field_name,
                   const std::function<f64(Tvec, Tvec)> pos_to_val,
                   const i32 offset) {
                    return self.template set_field_value_lambda<f64>(
                        field_name, pos_to_val, offset);
                },
                py::arg("field_name"),
                py::arg("pos_to_val"),
                py::arg("offset") = 0)
            .def(
                "set_field_value_lambda_f64_3",
                [](T &self,
                   std::string field_name,
                   const std::function<f64_3(Tvec, Tvec)> pos_to_val,
                   const i32 offset) {
                    return self.template set_field_value_lambda<f64_3>(
                        field_name, pos_to_val, offset);
                },
                py::arg("field_name"),
                py::arg("pos_to_val"),
                py::arg("offset") = 0)
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
                })
            .def(
                "get_solver_tex",
                [](T &self) {
                    return shambase::get_check_ref(self.solver.storage.solver_sequence).get_tex();
                })
            .def("get_solver_dot_graph", [](T &self) {
                return shambase::get_check_ref(self.solver.storage.solver_sequence).get_dot_graph();
            });
    }
} // namespace shammodels::basegodunov

Register_pymod(pybasegodunovmodel) {

    py::module mramses = m.def_submodule("model_ramses", "Shamrock Ramses solver");

    std::string base_name = "RamsesModel";
    using namespace shammodels::basegodunov;

    add_instance<f64_3, i64_3>(
        mramses, base_name + "_f64_3_i64_3_SolverConfig", base_name + "_f64_3_i64_3_Model");

    using VariantAMRGodunovBind = std::variant<std::unique_ptr<Model<f64_3, i64_3>>>;

    m.def(
        "get_Model_Ramses",
        [](ShamrockCtx &ctx,
           std::string vector_type,
           std::string grid_repr) -> VariantAMRGodunovBind {
            VariantAMRGodunovBind ret;

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
