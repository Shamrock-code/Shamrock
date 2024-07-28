// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyAMRGodunovModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include <memory>

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/amr/basegodunov/Model.hpp"
#include <pybind11/functional.h>

namespace shammodels::basegodunov {
    template<class Tvec, class TgridVec>
    void add_instance(py::module &m, std::string name_config, std::string name_model) {

        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        using T       = Model<Tvec, TgridVec>;
        using TConfig = typename T::Solver::Config;

        logger::debug_ln("[Py]", "registering class :",name_config,typeid(T).name());
        logger::debug_ln("[Py]", "registering class :",name_model,typeid(T).name());

        py::class_<TConfig>(m, name_config.c_str())
            .def("set_scale_factor",[](TConfig & self, Tscal scale_factor){
                self.grid_coord_to_pos_fact = scale_factor;
            })
            .def("set_Csafe",[](TConfig & self, Tscal Csafe){
                self.Csafe = Csafe;
            })
            .def("set_eos_gamma",[](TConfig & self, Tscal eos_gamma){
                self.set_eos_gamma(eos_gamma);
            })
            .def("set_riemann_solver_hll",[](TConfig & self){
                self.riemman_config = HLL;
            })
            .def("set_riemann_solver_rusanov",[](TConfig & self){
                self.riemman_config = Rusanov;
            })
            .def("set_slope_lim_none",[](TConfig & self){
                self.slope_config = None;
            })
            .def("set_slope_lim_vanleer_f",[](TConfig & self){
                self.slope_config = VanLeer_f;
            })
            .def("set_slope_lim_vanleer_std",[](TConfig & self){
                self.slope_config = VanLeer_std;
            })
            .def("set_slope_lim_vanleer_sym",[](TConfig & self){
                self.slope_config = VanLeer_sym;
            })
            .def("set_slope_lim_minmod",[](TConfig & self){
                self.slope_config = Minmod;
            });

        py::class_<T>(m, name_model.c_str())
            .def("init_scheduler", &T::init_scheduler)
            .def("make_base_grid", &T::make_base_grid)
            .def("dump_vtk", &T::dump_vtk)
            .def("evolve_once", &T::evolve_once)
            .def("set_field_value_lambda_f64",&T::template set_field_value_lambda<f64>)
            .def("set_field_value_lambda_f64_3",&T::template set_field_value_lambda<f64_3>)
            .def("gen_default_config",[](T & self) -> TConfig {
                return TConfig();
            })
            .def("set_config",[](T& self, TConfig cfg){
                self.solver.solver_config = cfg;
            })
            .def("get_cell_coords",[](T & self, std::pair<TgridVec,TgridVec> block_coord, u32 cell_local_id){
                return self.get_cell_coords(block_coord, cell_local_id);
            });
    }
}

Register_pymod(pybasegodunovmodel) {
    std::string base_name = "AMRGodunov";
    using namespace shammodels::basegodunov;

    add_instance<f64_3, i64_3>(
        m, base_name + "_f64_3_i64_3_SolverConfig", base_name + "_f64_3_i64_3_Model");

    using VariantAMRGodunovBind = std::variant<std::unique_ptr<Model<f64_3, i64_3>>>;

    m.def(
        "get_AMRGodunov",
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