// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySPHModel.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * \todo move to shambindings
 */
 
#include <memory>

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammath/sphkernels.hpp"

template<class Tvec, template<class> class SPHKernel>
void add_instance(py::module &m, std::string name_config, std::string name_model) {
    using namespace shammodels::sph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T       = Model<Tvec, SPHKernel>;
    using TConfig = typename T::Solver::Config;


    logger::debug_ln("[Py]", "registering class :",name_config,typeid(T).name());
    logger::debug_ln("[Py]", "registering class :",name_model,typeid(T).name());

    py::class_<TConfig>(m, name_config.c_str())
        .def("print_status", &TConfig::print_status)
        .def("set_artif_viscosity_None", &TConfig::set_artif_viscosity_None)
        .def(
            "set_artif_viscosity_Constant",
            [](TConfig &self, Tscal alpha_u, Tscal alpha_AV, Tscal beta_AV) {
                self.set_artif_viscosity_Constant({alpha_u, alpha_AV, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_u"),
            py::arg("alpha_AV"),
            py::arg("beta_AV"))
        .def(
            "set_artif_viscosity_VaryingMM97",
            [](TConfig &self,
               Tscal alpha_min,
               Tscal alpha_max,
               Tscal sigma_decay,
               Tscal alpha_u,
               Tscal beta_AV) {
                self.set_artif_viscosity_VaryingMM97(
                    {alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_min"),
            py::arg("alpha_max"),
            py::arg("sigma_decay"),
            py::arg("alpha_u"),
            py::arg("beta_AV"))
        .def(
            "set_artif_viscosity_VaryingCD10",
            [](TConfig &self,
               Tscal alpha_min,
               Tscal alpha_max,
               Tscal sigma_decay,
               Tscal alpha_u,
               Tscal beta_AV) {
                self.set_artif_viscosity_VaryingCD10(
                    {alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_min"),
            py::arg("alpha_max"),
            py::arg("sigma_decay"),
            py::arg("alpha_u"),
            py::arg("beta_AV"))
        .def("set_boundary_free",&TConfig::set_boundary_free)
        .def("set_boundary_periodic",&TConfig::set_boundary_periodic)
        .def("set_boundary_shearing_periodic",&TConfig::set_boundary_shearing_periodic)
        .def("set_units", &TConfig::set_units);

    py::class_<T>(m, name_model.c_str())
        .def(py::init([](ShamrockCtx &ctx) { return std::make_unique<T>(ctx); }))
        .def("init_scheduler", &T::init_scheduler)
        .def("evolve", &T::evolve_once)
        .def("set_cfl_cour", &T::set_cfl_cour)
        .def("set_cfl_force", &T::set_cfl_force)
        .def("set_particle_mass", &T::set_particle_mass)
        .def("set_eos_gamma", &T::set_eos_gamma)
        .def("rho_h", &T::rho_h)
        .def("get_hfact", &T::get_hfact)
        .def("get_box_dim_fcc_3d",
             [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
                 return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
             })
        .def("get_ideal_fcc_box",
             [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                 return self.get_ideal_fcc_box(dr, {box_min, box_max});
             })
        .def("resize_simulation_box",
             [](T &self, f64_3 box_min, f64_3 box_max) {
                 return self.resize_simulation_box({box_min, box_max});
             })
        .def("push_particle",
             [](T &self, std::vector<f64_3> pos, std::vector<f64> hpart, std::vector<f64> upart) {
                 return self.push_particle(pos,hpart, upart);
             })
        .def("add_cube_fcc_3d",
             [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                 return self.add_cube_fcc_3d(dr, {box_min, box_max});
             })
        .def("add_disc_3d_keplerian",
             [](T &self,
                Tvec center,
                u32 Npart,
                Tscal p,
                Tscal rho_0,
                Tscal m,
                Tscal r_in,
                Tscal r_out,
                Tscal q,
                Tscal cmass) {
                 return self.add_cube_disc_3d(center, Npart, p, rho_0, m, r_in, r_out, q,cmass);
             })
        .def("add_disc_3d",[](T &self,
                Tvec center, 
                Tscal central_mass,
                u32 Npart,
                Tscal r_in,
                Tscal r_out,
                Tscal disc_mass,
                Tscal p,
                Tscal H_r_in,
                Tscal q){
                    return self.add_disc_3d(center, central_mass, Npart, r_in, r_out, disc_mass, p, H_r_in, q);
                })
        .def("get_total_part_count", &T::get_total_part_count)
        .def("total_mass_to_part_mass", &T::total_mass_to_part_mass)
        .def("set_value_in_a_box",
             [](T &self,
                std::string field_name,
                std::string field_type,
                pybind11::object value,
                f64_3 box_min,
                f64_3 box_max) {
                 if (field_type == "f64") {
                     f64 val = value.cast<f64>();
                     self.set_value_in_a_box(field_name, val, {box_min, box_max});
                 } else if (field_type == "f64_3") {
                     f64_3 val = value.cast<f64_3>();
                     self.set_value_in_a_box(field_name, val, {box_min, box_max});
                 } else {
                     throw shambase::throw_with_loc<std::invalid_argument>("unknown field type");
                 }
             })
        .def("set_value_in_sphere",
             [](T &self,
                std::string field_name,
                std::string field_type,
                pybind11::object value,
                f64_3 center,
                f64 radius) {
                 if (field_type == "f64") {
                     f64 val = value.cast<f64>();
                     self.set_value_in_sphere(field_name, val, center, radius);
                 } else if (field_type == "f64_3") {
                     f64_3 val = value.cast<f64_3>();
                     self.set_value_in_sphere(field_name, val, center, radius);
                 } else {
                     throw shambase::throw_with_loc<std::invalid_argument>("unknown field type");
                 }
             })
        .def("set_field_value_lambda_f64_3",&T::template set_field_value_lambda<f64_3>)
        //.def("set_field_value_lambda_f64_3",[](T&self,std::string field_name, const std::function<f64_3 (Tscal, Tscal , Tscal)> pos_to_val){
        //    self.template set_field_value_lambda<f64_3>(field_name, [=](Tvec v){
        //        return pos_to_val(v.x(), v.y(),v.z());
        //    });
        //})
        .def("add_kernel_value",
             [](T &self,
                std::string field_name,
                std::string field_type,
                pybind11::object value,
                f64_3 center,
                f64 h_ker) {
                 if (field_type == "f64") {
                     f64 val = value.cast<f64>();
                     self.add_kernel_value(field_name, val, center, h_ker);
                 } else if (field_type == "f64_3") {
                     f64_3 val = value.cast<f64_3>();
                     self.add_kernel_value(field_name, val, center, h_ker);
                 } else {
                     throw shambase::throw_with_loc<std::invalid_argument>("unknown field type");
                 }
             })
        .def("get_sum",
             [](T &self, std::string field_name, std::string field_type) {
                 if (field_type == "f64") {
                     return py::cast(self.template get_sum<f64>(field_name));
                 } else if (field_type == "f64_3") {
                     return py::cast(self.template get_sum<f64_3>(field_name));
                 } else {
                     throw shambase::throw_with_loc<std::invalid_argument>("unknown field type");
                 }
             })
        .def("get_closest_part_to", [](T & self,f64_3 pos) -> f64_3 {
            return self.get_closest_part_to(pos);
        })
        .def("gen_default_config", [](T &self) { return typename T::Solver::Config{}; })
        .def("set_solver_config", &T::set_solver_config)
        .def("add_sink", &T::add_sink);
    ;
}

Register_pymod(pysphmodel) {

    using namespace shammodels::sph;

    add_instance<f64_3, shammath::M4>(
        m, "SPHModel_f64_3_M4_SolverConfig", "SPHModel_f64_3_M4");
    add_instance<f64_3, shammath::M6>(
        m, "SPHModel_f64_3_M6_SolverConfig", "SPHModel_f64_3_M6");

    using VariantSPHModelBind =
        std::variant<std::unique_ptr<Model<f64_3, shammath::M4>>,
                     std::unique_ptr<Model<f64_3, shammath::M6>>>;

    m.def(
        "get_SPHModel",
        [](ShamrockCtx &ctx, std::string vector_type, std::string kernel) -> VariantSPHModelBind {
            VariantSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M6") {
                ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
            } else {
                throw shambase::throw_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"));
}