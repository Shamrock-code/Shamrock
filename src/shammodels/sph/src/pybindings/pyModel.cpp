// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyModel.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::Model
 */

#include "shambase/memory.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/common/modules/ComputeGravWave.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/modules/AnalysisDisc.hpp"
#include "shammodels/sph/modules/AnalysisSodTube.hpp"
#include "shammodels/sph/modules/render/CartesianRender.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamphys/SodTube.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "sph_pybindings.hpp"
#include <pybind11/numpy.h>
#include <memory>
#include <optional>
#include <random>

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_model_instance(py::module &m, const std::string &name_model) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;

        using T = Model<Tvec, SPHKernel>;

        using TAnalysisSodTube = shammodels::sph::modules::AnalysisSodTube<Tvec, SPHKernel>;
        using TAnalysisDisc    = shammodels::sph::modules::AnalysisDisc<Tvec, SPHKernel>;

        using custom_getter_t
            = std::function<pybind11::array_t<f64>(size_t, shamrock::PatchDataLazyGetter &)>;

        py::class_<T> cls(m, name_model.c_str());
        cls.def(py::init([](ShamrockCtx &ctx) {
            return std::make_unique<T>(ctx);
        }));
        cls.def("init", &T::init);
        cls.def("init_scheduler", &T::init_scheduler);
        cls.def(
            "evolve_once_override_time",
            &T::evolve_once_time_expl,
            py::arg("t_curr"),
            py::arg("dt_input"));
        cls.def("evolve_once", &T::evolve_once);
        cls.def(
            "evolve_until",
            [](T &self, f64 target_time, i32 niter_max, f64 max_walltime) {
                return self.evolve_until(target_time, niter_max, max_walltime);
            },
            py::arg("target_time"),
            py::kw_only(),
            py::arg("niter_max")    = -1,
            py::arg("max_walltime") = -1);
        cls.def("timestep", &T::timestep);
        cls.def("set_cfl_cour", &T::set_cfl_cour, py::arg("cfl_cour"));
        cls.def("set_cfl_force", &T::set_cfl_force, py::arg("cfl_force"));
        cls.def("set_eta_sink", &T::set_eta_sink, py::arg("eta_sink"));
        cls.def("set_particle_mass", &T::set_particle_mass, py::arg("gpart_mass"));
        cls.def("get_particle_mass", &T::get_particle_mass);
        cls.def("rho_h", &T::rho_h);
        cls.def("get_hfact", &T::get_hfact);
        cls.def("get_solver_tex", [](T &self) {
            return shambase::get_check_ref(self.solver.storage.solver_sequence).get_tex();
        });
        cls.def("get_solver_dot_graph", [](T &self) {
            return shambase::get_check_ref(self.solver.storage.solver_sequence).get_dot_graph();
        });
        cls.def("get_box_dim_fcc_3d", [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
            return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
        });
        cls.def(
            "get_box_dim_true_fcc_3d",
            [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
                return self.get_box_dim_true_fcc_3d(dr, xcnt, ycnt, zcnt);
            },
            py::arg("dr"),
            py::arg("xcnt"),
            py::arg("ycnt"),
            py::arg("zcnt"),
            R"==(
    Get the dimensions of a periodic box holding a true FCC lattice of xcnt * ycnt * zcnt
    (see setup.make_generator_lattice_fcc). Unlike get_box_dim_fcc_3d (which actually
    describes a HCP lattice and is kept for backward compatibility), this throws if the box
    cannot be periodic: xcnt must be >= 2, ycnt even and zcnt a multiple of 3.
)==");
        cls.def("get_ideal_fcc_box", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            ON_RANK_0(
                shamcomm::logs::warn_ln(
                    "SPH",
                    "The python function get_ideal_fcc_box is deprecated in the SPH model and "
                    "will be removed at some point, replace it by "
                    "shamrock.math.get_ideal_hcp_box"));
            return shammath::LatticeHCP<f64_3>::get_ideal_hcp_box(dr, {box_min, box_max});
        });
        cls.def("get_ideal_hcp_box", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            ON_RANK_0(
                shamcomm::logs::warn_ln(
                    "SPH",
                    "The python function get_ideal_hcp_box is deprecated in the SPH model and "
                    "will be removed at some point, replace it by "
                    "shamrock.math.get_ideal_hcp_box"));
            return shammath::LatticeHCP<f64_3>::get_ideal_hcp_box(dr, {box_min, box_max});
        });
        cls.def("resize_simulation_box", [](T &self, f64_3 box_min, f64_3 box_max) {
            return self.resize_simulation_box({box_min, box_max});
        });
        cls.def(
            "push_particle",
            [](T &self, std::vector<f64_3> pos, std::vector<f64> hpart, std::vector<f64> upart) {
                return self.push_particle(pos, hpart, upart);
            });
        cls.def(
            "push_particle_mhd",
            [](T &self,
               std::vector<f64_3> pos,
               std::vector<f64> hpart,
               std::vector<f64> upart,
               std::vector<f64_3> B_on_rho,
               std::vector<f64> psi_on_ch) {
                return self.push_particle_mhd(pos, hpart, upart, B_on_rho, psi_on_ch);
            });
        cls.def("add_cube_fcc_3d", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.add_cube_fcc_3d(dr, {box_min, box_max});
        });
        cls.def("add_cube_hcp_3d", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.add_cube_hcp_3d(dr, {box_min, box_max});
        });
        cls.def("add_cube_hcp_3d_v2", [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
            return self.add_cube_hcp_3d_v2(dr, {box_min, box_max});
        });
        cls.def(
            "add_disc_3d_keplerian",
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
                return self.add_cube_disc_3d(center, Npart, p, rho_0, m, r_in, r_out, q, cmass);
            });
        cls.def(
            "add_disc_3d",
            [](T &self,
               Tvec center,
               Tscal central_mass,
               u32 Npart,
               Tscal r_in,
               Tscal r_out,
               Tscal disc_mass,
               Tscal p,
               Tscal H_r_in,
               Tscal q) {
                return self.add_disc_3d(
                    center, central_mass, Npart, r_in, r_out, disc_mass, p, H_r_in, q);
            });
        cls.def(
            "add_big_disc_3d",
            [](T &self,
               Tvec center,
               Tscal central_mass,
               u32 Npart,
               Tscal r_in,
               Tscal r_out,
               Tscal disc_mass,
               Tscal p,
               Tscal H_r_in,
               Tscal q,
               u16 seed) {
                self.add_big_disc_3d(
                    center,
                    central_mass,
                    Npart,
                    r_in,
                    r_out,
                    disc_mass,
                    p,
                    H_r_in,
                    q,
                    std::mt19937{seed});
                return disc_mass / Npart;
            });
        cls.def("get_total_part_count", &T::get_total_part_count);
        cls.def("total_mass_to_part_mass", &T::total_mass_to_part_mass);
        cls.def(
            "set_value_in_a_box",
            [](T &self,
               const std::string &field_name,
               const std::string &field_type,
               const pybind11::object &value,
               f64_3 box_min,
               f64_3 box_max,
               u32 ivar) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            },
            py::arg("field_name"),
            py::arg("field_type"),
            py::arg("value"),
            py::arg("box_min"),
            py::arg("box_max"),
            py::kw_only(),
            py::arg("ivar") = 0);
        cls.def(
            "set_value_in_sphere",
            [](T &self,
               const std::string &field_name,
               const std::string &field_type,
               const pybind11::object &value,
               f64_3 center,
               f64 radius) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_value_in_sphere(field_name, val, center, radius);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_value_in_sphere(field_name, val, center, radius);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            });
        cls.def(
            "set_field_value_lambda_f64",
            [](T &self,
               std::string field_name,
               const std::function<f64(Tvec)> pos_to_val,
               const u32 offset) {
                return self.template set_field_value_lambda<f64>(
                    std::move(field_name), pos_to_val, offset);
            },
            py::arg("field_name"),
            py::arg("pos_to_val"),
            py::arg("offset") = 0);
        cls.def(
            "set_field_value_lambda_f64_3",
            [](T &self,
               std::string field_name,
               const std::function<f64_3(Tvec)> pos_to_val,
               const u32 offset) {
                return self.template set_field_value_lambda<f64_3>(
                    std::move(field_name), pos_to_val, offset);
            },
            py::arg("field_name"),
            py::arg("pos_to_val"),
            py::arg("offset") = 0);
        cls.def("overwrite_field_value_f64", &T::template overwrite_field_value<f64>);
        cls.def("overwrite_field_value_f64_3", &T::template overwrite_field_value<f64_3>);
        cls.def("remap_positions", &T::remap_positions);
        //.def("set_field_value_lambda_f64_3",[](T&self,std::string field_name, const
        // std::function<f64_3 (Tscal, Tscal , Tscal)> pos_to_val){
        //    self.template set_field_value_lambda<f64_3>(field_name, [=](Tvec v){
        //        return pos_to_val(v.x(), v.y(),v.z());
        //    });
        //})
        cls.def(
            "add_kernel_value",
            [](T &self,
               const std::string &field_name,
               const std::string &field_type,
               const pybind11::object &value,
               f64_3 center,
               f64 h_ker) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.add_kernel_value(field_name, val, center, h_ker);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.add_kernel_value(field_name, val, center, h_ker);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            });
        cls.def(
            "get_sum", [](T &self, const std::string &field_name, const std::string &field_type) {
                if (field_type == "f64") {
                    return py::cast(self.template get_sum<f64>(field_name));
                } else if (field_type == "f64_3") {
                    return py::cast(self.template get_sum<f64_3>(field_name));
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            });
        cls.def("get_closest_part_to", [](T &self, f64_3 pos) -> f64_3 {
            return self.get_closest_part_to(pos);
        });
        cls.def("gen_default_config", [](T &self) {
            return typename T::Solver::Config{};
        });
        cls.def("get_current_config", [](T &self) {
            return self.solver.solver_config;
        });
        cls.def("set_solver_config", &T::set_solver_config);
        cls.def("add_sink", &T::add_sink);
        cls.def("get_sinks", [](T &self) {
            py::list list_out;

            auto &sync = shambase::get_check_ref(self.ctx.sched).synchronized_data;
            auto edges = get_sink_edges<Tvec>(sync);
            for (auto &sink : to_sink_particles(edges)) {
                py::dict sink_dic;
                sink_dic["pos"]              = sink.pos;
                sink_dic["velocity"]         = sink.velocity;
                sink_dic["sph_acceleration"] = sink.sph_acceleration;
                sink_dic["ext_acceleration"] = sink.ext_acceleration;
                sink_dic["mass"]             = sink.mass;
                sink_dic["angular_momentum"] = sink.angular_momentum;
                sink_dic["accretion_radius"] = sink.accretion_radius;
                list_out.append(sink_dic);
            }

            // only present with dust, flattened as [i_sink * ndust + j_dust]
            if (has_sink_accreted_dust_mass(sync)) {
                auto &dust_mass = get_sink_accreted_dust_mass<Tvec>(sync);
                size_t nsink    = list_out.size();
                size_t ndust    = (nsink == 0) ? 0 : dust_mass.size() / nsink;
                for (size_t i = 0; i < nsink; i++) {
                    auto begin                        = dust_mass.begin() + i * ndust;
                    list_out[i]["accreted_dust_mass"] = std::vector<Tscal>(begin, begin + ndust);
                }
            }

            return list_out;
        });
        cls.def("get_units", [](T &self) {
            return self.solver.solver_config.unit_sys;
        });
        cls.def(
            "render_slice",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               const std::vector<Tvec> &positions,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_slice(name, positions, custom_getter).copy_to_stdvec();
                }

                if (field_type == "f64_3") {
                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_slice(name, positions, std::nullopt).copy_to_stdvec();
                }

                throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("positions"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_slice",
            [](T &self,
               shamrock::solvergraph::Field<f64> &field,
               const std::vector<Tvec> &positions) -> std::vector<f64> {
                modules::CartesianRender<Tvec, f64, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_slice(field, positions).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("positions"));
        cls.def(
            "render_slice",
            [](T &self,
               shamrock::solvergraph::Field<f64_3> &field,
               const std::vector<Tvec> &positions) -> std::vector<f64_3> {
                modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_slice(field, positions).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("positions"));
        cls.def(
            "render_column_integ",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               const std::vector<shammath::Ray<Tvec>> &rays,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_column_integ(name, rays, custom_getter).copy_to_stdvec();
                }

                if (field_type == "f64_3") {
                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_column_integ(name, rays, std::nullopt).copy_to_stdvec();
                }

                throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("rays"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_column_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64> &field,
               const std::vector<shammath::Ray<Tvec>> &rays) -> std::vector<f64> {
                modules::CartesianRender<Tvec, f64, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_column_integ(field, rays).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("rays"));
        cls.def(
            "render_column_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64_3> &field,
               const std::vector<shammath::Ray<Tvec>> &rays) -> std::vector<f64_3> {
                modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_column_integ(field, rays).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("rays"));
        cls.def(
            "compute_field",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<
                    shamrock::solvergraph::Field<f64>,
                    shamrock::solvergraph::Field<f64_3>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    modules::RenderFieldGetter<Tvec, f64, SPHKernel> render_field_getter(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render_field_getter.build_field(name, custom_getter);
                }

                if (field_type == "f64_3") {
                    modules::RenderFieldGetter<Tvec, f64_3, SPHKernel> render_field_getter(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render_field_getter.build_field(name, custom_getter);
                }

                throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_azymuthal_integ",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               const std::vector<shammath::RingRay<Tvec>> &ring_rays,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<std::vector<f64>, std::vector<f64_3>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_azymuthal_integ(name, ring_rays, custom_getter)
                        .copy_to_stdvec();
                }

                if (field_type == "f64_3") {
                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);
                    return render.compute_azymuthal_integ(name, ring_rays, std::nullopt)
                        .copy_to_stdvec();
                }

                throw shambase::make_except_with_loc<std::runtime_error>("unknown field type");
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("ring_rays"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_azymuthal_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64> &field,
               const std::vector<shammath::RingRay<Tvec>> &ring_rays) -> std::vector<f64> {
                modules::CartesianRender<Tvec, f64, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_azymuthal_integ(field, ring_rays).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("ring_rays"));
        cls.def(
            "render_azymuthal_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64_3> &field,
               const std::vector<shammath::RingRay<Tvec>> &ring_rays) -> std::vector<f64_3> {
                modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);
                return render.compute_azymuthal_integ(field, ring_rays).copy_to_stdvec();
            },
            py::arg("field"),
            py::arg("ring_rays"));
        cls.def(
            "render_cartesian_slice",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<py::array_t<Tscal>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    py::array_t<Tscal> ret({ny, nx});

                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64> slice
                        = render
                              .compute_slice(name, center, delta_x, delta_y, nx, ny, custom_getter)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                        }
                    }

                    return ret;
                }

                if (field_type == "f64_3") {
                    py::array_t<Tscal> ret({ny, nx, 3_u32});

                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64_3> slice
                        = render.compute_slice(name, center, delta_x, delta_y, nx, ny, std::nullopt)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                            ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                            ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                        }
                    }

                    return ret;
                }

                shambase::throw_with_loc<std::runtime_error>("unknown field type");
                return py::array_t<Tscal>({nx, ny});
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_cartesian_slice",
            [](T &self,
               shamrock::solvergraph::Field<f64> &field,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> py::array_t<Tscal> {
                py::array_t<Tscal> ret({ny, nx});

                modules::CartesianRender<Tvec, f64, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);

                std::vector<f64> slice
                    = render.compute_slice(field, center, delta_x, delta_y, nx, ny)
                          .copy_to_stdvec();

                for (u32 iy = 0; iy < ny; iy++) {
                    for (u32 ix = 0; ix < nx; ix++) {
                        ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                    }
                }

                return ret;
            },
            py::arg("field"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"));
        cls.def(
            "render_cartesian_slice",
            [](T &self,
               shamrock::solvergraph::Field<f64_3> &field,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> py::array_t<Tscal> {
                py::array_t<Tscal> ret({ny, nx, 3_u32});

                modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);

                std::vector<f64_3> slice
                    = render.compute_slice(field, center, delta_x, delta_y, nx, ny)
                          .copy_to_stdvec();

                for (u32 iy = 0; iy < ny; iy++) {
                    for (u32 ix = 0; ix < nx; ix++) {
                        ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                        ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                        ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                    }
                }

                return ret;
            },
            py::arg("field"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"));
        cls.def(
            "render_cartesian_column_integ",
            [](T &self,
               const std::string &name,
               const std::string &field_type,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny,
               const std::optional<custom_getter_t> &custom_getter)
                -> std::variant<py::array_t<Tscal>> {
                if (custom_getter.has_value()) {
                    if (!(name == "custom" && field_type == "f64")) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_getter only available for name=custom and field_type=f64");
                    }
                }

                if (field_type == "f64") {
                    py::array_t<Tscal> ret({ny, nx});

                    modules::CartesianRender<Tvec, f64, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64> slice
                        = render
                              .compute_column_integ(
                                  name, center, delta_x, delta_y, nx, ny, custom_getter)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                        }
                    }

                    return ret;
                }

                if (field_type == "f64_3") {
                    py::array_t<Tscal> ret({ny, nx, 3_u32});

                    modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                        self.ctx, self.solver.solver_config, self.solver.storage);

                    std::vector<f64_3> slice
                        = render
                              .compute_column_integ(
                                  name, center, delta_x, delta_y, nx, ny, std::nullopt)
                              .copy_to_stdvec();

                    for (u32 iy = 0; iy < ny; iy++) {
                        for (u32 ix = 0; ix < nx; ix++) {
                            ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                            ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                            ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                        }
                    }

                    return ret;
                }

                shambase::throw_with_loc<std::runtime_error>("unknown field type");
                return py::array_t<Tscal>({nx, ny});
            },
            py::arg("name"),
            py::arg("field_type"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("custom_getter") = std::nullopt);
        cls.def(
            "render_cartesian_column_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64> &field,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> py::array_t<Tscal> {
                py::array_t<Tscal> ret({ny, nx});

                modules::CartesianRender<Tvec, f64, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);

                std::vector<f64> slice
                    = render.compute_column_integ(field, center, delta_x, delta_y, nx, ny)
                          .copy_to_stdvec();

                for (u32 iy = 0; iy < ny; iy++) {
                    for (u32 ix = 0; ix < nx; ix++) {
                        ret.mutable_at(iy, ix) = slice[ix + nx * iy];
                    }
                }

                return ret;
            },
            py::arg("field"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"));
        cls.def(
            "render_cartesian_column_integ",
            [](T &self,
               shamrock::solvergraph::Field<f64_3> &field,
               Tvec center,
               Tvec delta_x,
               Tvec delta_y,
               u32 nx,
               u32 ny) -> py::array_t<Tscal> {
                py::array_t<Tscal> ret({ny, nx, 3_u32});

                modules::CartesianRender<Tvec, f64_3, SPHKernel> render(
                    self.ctx, self.solver.solver_config, self.solver.storage);

                std::vector<f64_3> slice
                    = render.compute_column_integ(field, center, delta_x, delta_y, nx, ny)
                          .copy_to_stdvec();

                for (u32 iy = 0; iy < ny; iy++) {
                    for (u32 ix = 0; ix < nx; ix++) {
                        ret.mutable_at(iy, ix, 0) = slice[ix + nx * iy][0];
                        ret.mutable_at(iy, ix, 1) = slice[ix + nx * iy][1];
                        ret.mutable_at(iy, ix, 2) = slice[ix + nx * iy][2];
                    }
                }

                return ret;
            },
            py::arg("field"),
            py::arg("center"),
            py::arg("delta_x"),
            py::arg("delta_y"),
            py::arg("nx"),
            py::arg("ny"));
        cls.def(
            "gen_config_from_phantom_dump",
            [](T &self, PhantomDump &dump, bool bypass_error) {
                return self.gen_config_from_phantom_dump(dump, bypass_error);
            },
            py::arg("dump"),
            py::arg("bypass_error") = false,
            R"==(
    This function generate a shamrock sph solver config from a phantom dump

    Parameters
    ----------
    PhantomDump dump
    bypass_error = false (default) bypass any error in the config
)==");
        cls.def(
            "init_from_phantom_dump",
            [](T &self, PhantomDump &dump, Tscal hpart_fact_load) {
                self.init_from_phantom_dump(dump, hpart_fact_load);
            },
            py::arg("dump"),
            py::arg("hpart_fact_load") = 1.0);
        cls.def("make_phantom_dump", [](T &self) {
            return self.make_phantom_dump();
        });
        cls.def("do_vtk_dump", &T::do_vtk_dump);
        cls.def("set_debug_dump", &T::set_debug_dump);
        cls.def("solver_logs_last_rate", &T::solver_logs_last_rate);
        cls.def("solver_logs_last_obj_count", &T::solver_logs_last_obj_count);
        cls.def("solver_logs_last_system_metrics", [&](T &self) {
            auto system_metrics = self.solver.solve_logs.get_last_system_metrics();
            py::dict ret;
            ret["duration"] = system_metrics.wall_time;
            if (system_metrics.rank_energy_consummed.has_value()) {
                ret["rank_energy_consummed"] = system_metrics.rank_energy_consummed.value();
            }
            if (system_metrics.gpu_energy_consummed.has_value()) {
                ret["gpu_energy_consummed"] = system_metrics.gpu_energy_consummed.value();
            }
            if (system_metrics.cpu_energy_consummed.has_value()) {
                ret["cpu_energy_consummed"] = system_metrics.cpu_energy_consummed.value();
            }
            if (system_metrics.dram_energy_consummed.has_value()) {
                ret["dram_energy_consummed"] = system_metrics.dram_energy_consummed.value();
            }
            return ret;
        });
        cls.def("solver_logs_cumulated_step_time", &T::solver_logs_cumulated_step_time);
        cls.def("solver_logs_reset_cumulated_step_time", &T::solver_logs_reset_cumulated_step_time);
        cls.def("solver_logs_step_count", &T::solver_logs_step_count);
        cls.def("solver_logs_reset_step_count", &T::solver_logs_reset_step_count);
        cls.def("get_time", [](T &self) {
            return self.get_time();
        });
        cls.def("get_dt", [](T &self) {
            return self.get_dt_sph();
        });
        cls.def(
            "get_last_cfl_dts",
            [](T &self) {
                py::dict ret;
                for (auto &[name, value] : self.solver.get_last_cfl_dts()) {
                    ret[py::str(name)] = value;
                }
                return ret;
            },
            R"==(
    Get the last CFL timesteps stored in the synchronized edges.

    Returns a dict keyed by synchronized edge name:

    - ``"cfl_dt"``: the global CFL timestep of the last step (min over all CFL conditions).
    - ``"cfl_dt_<type>"`` (e.g. ``"cfl_dt_courant"``, ``"cfl_dt_force"``,
      ``"cfl_dt_sink_sink"``): the CFL timestep of each condition enabled in the solver
      config, only present when ``show_cfl_detail`` is enabled.

    All values are 0 before the first step. A condition that was not evaluated during the
    last step (e.g. ``"cfl_dt_sink_sink"`` with no sinks) keeps its previous value.
)==");
        cls.def("set_time", [](T &self, Tscal t) {
            return self.set_time(t);
        });
        cls.def("set_next_dt", [](T &self, Tscal dt) {
            return self.set_next_dt(dt);
        });
        cls.def("set_dt", [](T &self, f64 dt) {
            self.set_next_dt(dt);
        });
        cls.def(
            "set_cfl_multipler",
            [](T &self, Tscal lambda) {
                return self.set_cfl_multipler(lambda);
            },
            py::arg("lambda"));
        cls.def(
            "set_cfl_mult_stiffness",
            [](T &self, Tscal cstiff) {
                return self.solver.solver_config.set_cfl_mult_stiffness(cstiff);
            },
            py::arg("cstiff"));
        cls.def("change_htolerance", [](T &self, Tscal in) {
            ON_RANK_0(shamlog_warn_ln(
                          "SPH",
                          ".change_htolerance(val) is deprecated,\n"
                          "    -> calling this is replaced internally by "
                          ".change_htolerances(coarse=val, fine=min(val, 1.1))\n"
                          "    see: "
                          "https://shamrock-code.github.io/Shamrock/sphinx/user_guide/sph/"
                          "smoothing_length_tolerance.html"););
            self.change_htolerances(in, std::min(in, (Tscal) 1.1));
        });
        cls.def(
            "change_htolerances",
            [](T &self, Tscal coarse, Tscal fine) {
                self.change_htolerances(coarse, fine);
            },
            py::kw_only(),
            py::arg("coarse"),
            py::arg("fine"));
        cls.def(
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
            },
            py::arg("sod"),
            py::arg("direction"),
            py::arg("time_val"),
            py::arg("x_ref"),
            py::arg("x_min"),
            py::arg("x_max"));
        cls.def("make_analysis_disc", [](T &self) {
            return std::make_unique<TAnalysisDisc>(
                self.ctx, self.solver.solver_config, self.solver.storage);
        });
        cls.def("load_from_dump", &T::load_from_dump);
        cls.def("dump", &T::dump);
        cls.def("get_setup", &T::get_setup);
        cls.def("get_patch_transform", [](T &self) {
            PatchScheduler &sched = shambase::get_check_ref(self.ctx.sched);
            return sched.get_patch_transform<Tvec>();
        });
        cls.def("apply_momentum_offset", &T::apply_momentum_offset);
        cls.def("apply_position_offset", &T::apply_position_offset);
        cls.def(
            "add_timestep_callback",
            [](T &self,
               std::optional<std::function<void(void)>> step_begin_callback,
               std::optional<std::function<void(void)>> step_end_callback) {
                self.solver.timestep_callbacks.push_back(
                    {std::move(step_begin_callback), std::move(step_end_callback)});
            },
            py::kw_only(),
            py::arg("step_begin") = std::nullopt,
            py::arg("step_end")   = std::nullopt);
    }

    template void add_model_instance<f64_3, shammath::M4>(py::module &, const std::string &);
    template void add_model_instance<f64_3, shammath::M6>(py::module &, const std::string &);
    template void add_model_instance<f64_3, shammath::M8>(py::module &, const std::string &);
    template void add_model_instance<f64_3, shammath::C2>(py::module &, const std::string &);
    template void add_model_instance<f64_3, shammath::C4>(py::module &, const std::string &);
    template void add_model_instance<f64_3, shammath::C6>(py::module &, const std::string &);

} // namespace shammodels::sph::pybindings
