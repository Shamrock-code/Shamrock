// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyGSPHModel.cpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for the GSPH (Godunov SPH) model
 *
 * This provides Python interface for GSPH simulations using Riemann solvers.
 *
 * References:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/gsph/Model.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <memory>

template<class Tvec, template<class> class SPHKernel>
void add_gsph_instance(py::module &m, std::string name_config, std::string name_model) {
    using namespace shammodels::gsph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T       = Model<Tvec, SPHKernel>;
    using TConfig = typename T::SolverConfig;

    shamlog_debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
    shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

    py::class_<TConfig>(m, name_config.c_str())
        .def("print_status", &TConfig::print_status)
        .def("set_tree_reduction_level", &TConfig::set_tree_reduction_level)
        .def("set_two_stage_search", &TConfig::set_two_stage_search)
        // Riemann solver config
        .def(
            "set_riemann_iterative",
            [](TConfig &self, Tscal tol, u32 max_iter) {
                self.set_riemann_iterative(tol, max_iter);
            },
            py::kw_only(),
            py::arg("tolerance") = Tscal{1e-6},
            py::arg("max_iter")  = 20,
            R"==(
    Set iterative Riemann solver (van Leer 1997).

    This is the most accurate but slower Riemann solver.
    Uses Newton-Raphson iteration to find the pressure in the star region.

    Parameters
    ----------
    tolerance : float
        Convergence tolerance for Newton-Raphson iteration (default: 1e-6)
    max_iter : int
        Maximum number of iterations (default: 20)
)==")
        .def(
            "set_riemann_hllc",
            [](TConfig &self) {
                self.set_riemann_hllc();
            },
            R"==(
    Set HLLC approximate Riemann solver.

    Fast approximate Riemann solver that captures contact discontinuities.
    Recommended for general use - good balance of accuracy and speed.
)==")
        .def(
            "set_riemann_exact",
            [](TConfig &self, Tscal tol) {
                self.set_riemann_exact(tol);
            },
            py::kw_only(),
            py::arg("tolerance") = Tscal{1e-8},
            R"==(
    Set exact Riemann solver.

    Uses Newton iteration to find the exact solution.
    Most accurate but slowest.

    Parameters
    ----------
    tolerance : float
        Convergence tolerance (default: 1e-8)
)==")
        .def(
            "set_riemann_roe",
            [](TConfig &self, Tscal entropy_fix) {
                self.set_riemann_roe(entropy_fix);
            },
            py::kw_only(),
            py::arg("entropy_fix") = Tscal{0.1},
            R"==(
    Set Roe linearized Riemann solver.

    Fast approximate solver using linearization about the Roe average.
    May produce entropy-violating solutions near sonic points without entropy fix.

    Parameters
    ----------
    entropy_fix : float
        Entropy fix parameter (default: 0.1)
)==")
        // Reconstruction config
        .def(
            "set_reconstruct_piecewise_constant",
            [](TConfig &self) {
                self.set_reconstruct_piecewise_constant();
            },
            R"==(
    Set first-order piecewise constant reconstruction.

    Most diffusive but most stable. Good for initial testing.
)==")
        .def(
            "set_reconstruct_muscl",
            [](TConfig &self, std::string limiter) {
                using ReconstructConfig = typename TConfig::ReconstructConfig;
                typename ReconstructConfig::Limiter lim;

                if (limiter == "van_leer" || limiter == "VanLeer") {
                    lim = ReconstructConfig::Limiter::VanLeer;
                } else if (limiter == "minmod" || limiter == "Minmod") {
                    lim = ReconstructConfig::Limiter::Minmod;
                } else if (limiter == "superbee" || limiter == "Superbee") {
                    lim = ReconstructConfig::Limiter::Superbee;
                } else if (limiter == "mc" || limiter == "MC") {
                    lim = ReconstructConfig::Limiter::MC;
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "Unknown limiter: " + limiter
                        + ". Valid options: van_leer, minmod, superbee, mc");
                }

                self.set_reconstruct_muscl(lim);
            },
            py::kw_only(),
            py::arg("limiter") = "van_leer",
            R"==(
    Set second-order MUSCL reconstruction with slope limiting.

    More accurate than piecewise constant but may be less stable for
    strong discontinuities.

    Parameters
    ----------
    limiter : str
        Slope limiter type. Options: "van_leer" (default), "minmod", "superbee", "mc"
)==")
        // EOS config
        .def(
            "set_eos_adiabatic",
            [](TConfig &self, Tscal gamma) {
                self.set_eos_adiabatic(gamma);
            },
            py::arg("gamma"),
            R"==(
    Set adiabatic equation of state: P = (γ-1) × ρ × u

    Parameters
    ----------
    gamma : float
        Adiabatic index (e.g., 5/3 for monatomic gas, 7/5 for diatomic)
)==")
        .def(
            "set_eos_isothermal",
            [](TConfig &self, Tscal cs) {
                self.set_eos_isothermal(cs);
            },
            py::arg("cs"),
            R"==(
    Set isothermal equation of state: P = cs² × ρ

    Parameters
    ----------
    cs : float
        Sound speed
)==")
        // Boundary config
        .def("set_boundary_free", &TConfig::set_boundary_free)
        .def("set_boundary_periodic", &TConfig::set_boundary_periodic)
        .def(
            "set_boundary_wall",
            &TConfig::set_boundary_wall,
            py::arg("num_layers") = 4,
            py::arg("wall_flags") = 0x3F,
            R"==(
Set wall boundary condition.

Creates "wall particles" beyond domain boundaries that mirror boundary particles.
This provides proper neighbor support for particles near walls.

Parameters
----------
num_layers : int
    Number of wall particle layers beyond each boundary (default: 4)
wall_flags : int
    Bit flags for which walls to enable (default: 0x3F = all walls)
    Bit 0: -x, Bit 1: +x, Bit 2: -y, Bit 3: +y, Bit 4: -z, Bit 5: +z
)==")
        // External forces
        .def(
            "add_ext_force_point_mass",
            [](TConfig &self, Tscal central_mass, Tscal Racc) {
                self.add_ext_force_point_mass(central_mass, Racc);
            },
            py::kw_only(),
            py::arg("central_mass"),
            py::arg("Racc"))
        // Units
        .def("set_units", &TConfig::set_units)
        // CFL
        .def(
            "set_cfl_cour",
            [](TConfig &self, Tscal cfl_cour) {
                self.cfl_config.cfl_cour = cfl_cour;
            })
        .def(
            "set_cfl_force",
            [](TConfig &self, Tscal cfl_force) {
                self.cfl_config.cfl_force = cfl_force;
            })
        .def(
            "set_particle_mass",
            [](TConfig &self, Tscal gpart_mass) {
                self.gpart_mass = gpart_mass;
            })
        .def("to_json", [](TConfig &self) {
            return nlohmann::json{self}.dump(4);
        });

    py::class_<T>(m, name_model.c_str())
        .def(py::init([](ShamrockCtx &ctx) {
            return std::make_unique<T>(ctx);
        }))
        .def("init_scheduler", &T::init_scheduler)
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
        .def("set_cfl_cour", &T::set_cfl_cour, py::arg("cfl_cour"))
        .def("set_cfl_force", &T::set_cfl_force, py::arg("cfl_force"))
        .def("set_particle_mass", &T::set_particle_mass, py::arg("gpart_mass"))
        .def("get_particle_mass", &T::get_particle_mass)
        .def("rho_h", &T::rho_h)
        .def("get_hfact", &T::get_hfact)
        .def(
            "get_box_dim_fcc_3d",
            [](T &self, f64 dr, u32 xcnt, u32 ycnt, u32 zcnt) {
                return self.get_box_dim_fcc_3d(dr, xcnt, ycnt, zcnt);
            })
        .def(
            "get_ideal_fcc_box",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.get_ideal_fcc_box(dr, {box_min, box_max});
            })
        .def(
            "get_ideal_hcp_box",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.get_ideal_hcp_box(dr, {box_min, box_max});
            })
        .def(
            "resize_simulation_box",
            [](T &self, f64_3 box_min, f64_3 box_max) {
                return self.resize_simulation_box({box_min, box_max});
            })
        .def(
            "add_cube_fcc_3d",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.add_cube_fcc_3d(dr, {box_min, box_max});
            })
        .def(
            "add_cube_hcp_3d",
            [](T &self, f64 dr, f64_3 box_min, f64_3 box_max) {
                return self.add_cube_hcp_3d(dr, {box_min, box_max});
            })
        .def("get_total_part_count", &T::get_total_part_count)
        .def("total_mass_to_part_mass", &T::total_mass_to_part_mass)
        .def(
            "set_value_in_a_box",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 box_min,
               f64_3 box_max,
               u32 ivar) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                } else if (field_type == "u32") {
                    u32 val = value.cast<u32>();
                    self.set_value_in_a_box(field_name, val, {box_min, box_max}, ivar);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type: " + field_type + ". Valid types: f64, f64_3, u32");
                }
            },
            py::arg("field_name"),
            py::arg("field_type"),
            py::arg("value"),
            py::arg("box_min"),
            py::arg("box_max"),
            py::kw_only(),
            py::arg("ivar") = 0,
            R"==(
    Set field value for particles within a box region.

    Parameters
    ----------
    field_name : str
        Name of the field to set (e.g., "vxyz", "uint", "wall_flag")
    field_type : str
        Type of the field: "f64", "f64_3", or "u32"
    value : float, tuple, or int
        Value to set (type must match field_type)
    box_min : tuple
        Minimum corner of the box (x, y, z)
    box_max : tuple
        Maximum corner of the box (x, y, z)
    ivar : int
        Variable index for multi-component fields (default: 0)

    Examples
    --------
    >>> # Set wall_flag=1 for particles in a region
    >>> model.set_value_in_a_box("wall_flag", "u32", 1, (-1,-1,-1), (0,1,1))
)==")
        .def(
            "set_value_in_sphere",
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
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def("set_field_value_lambda_f64_3", &T::template set_field_value_lambda<f64_3>)
        .def("set_field_value_lambda_f64", &T::template set_field_value_lambda<f64>)
        .def(
            "get_sum",
            [](T &self, std::string field_name, std::string field_type) {
                if (field_type == "f64") {
                    return py::cast(self.template get_sum<f64>(field_name));
                } else if (field_type == "f64_3") {
                    return py::cast(self.template get_sum<f64_3>(field_name));
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            })
        .def(
            "gen_default_config",
            [](T &self) {
                return self.gen_default_config();
            })
        .def(
            "get_current_config",
            [](T &self) {
                return self.solver.solver_config;
            })
        .def("set_solver_config", &T::set_solver_config)
        .def("do_vtk_dump", &T::do_vtk_dump)
        .def("solver_logs_last_rate", &T::solver_logs_last_rate)
        .def("solver_logs_last_obj_count", &T::solver_logs_last_obj_count)
        .def(
            "get_time",
            [](T &self) {
                return self.solver.solver_config.get_time();
            })
        .def(
            "get_dt",
            [](T &self) {
                return self.solver.solver_config.get_dt();
            })
        .def(
            "set_time",
            [](T &self, Tscal t) {
                return self.solver.solver_config.set_time(t);
            })
        .def(
            "set_next_dt",
            [](T &self, Tscal dt) {
                return self.solver.solver_config.set_next_dt(dt);
            })
        .def("load_from_dump", &T::load_from_dump)
        .def("dump", &T::dump);
}

using namespace shammodels::gsph;

Register_pymod(pygsphmodel) {

    py::module mgsph = m.def_submodule("model_gsph", "Shamrock GSPH (Godunov SPH) solver");

    using namespace shammodels::gsph;

    // Register GSPH models for different kernels
    add_gsph_instance<f64_3, shammath::M4>(
        mgsph, "GSPHModel_f64_3_M4_SolverConfig", "GSPHModel_f64_3_M4");
    add_gsph_instance<f64_3, shammath::M6>(
        mgsph, "GSPHModel_f64_3_M6_SolverConfig", "GSPHModel_f64_3_M6");
    add_gsph_instance<f64_3, shammath::M8>(
        mgsph, "GSPHModel_f64_3_M8_SolverConfig", "GSPHModel_f64_3_M8");

    add_gsph_instance<f64_3, shammath::C2>(
        mgsph, "GSPHModel_f64_3_C2_SolverConfig", "GSPHModel_f64_3_C2");
    add_gsph_instance<f64_3, shammath::C4>(
        mgsph, "GSPHModel_f64_3_C4_SolverConfig", "GSPHModel_f64_3_C4");
    add_gsph_instance<f64_3, shammath::C6>(
        mgsph, "GSPHModel_f64_3_C6_SolverConfig", "GSPHModel_f64_3_C6");

    using VariantGSPHModelBind = std::variant<
        std::unique_ptr<Model<f64_3, shammath::M4>>,
        std::unique_ptr<Model<f64_3, shammath::M6>>,
        std::unique_ptr<Model<f64_3, shammath::M8>>,
        std::unique_ptr<Model<f64_3, shammath::C2>>,
        std::unique_ptr<Model<f64_3, shammath::C4>>,
        std::unique_ptr<Model<f64_3, shammath::C6>>>;

    m.def(
        "get_Model_GSPH",
        [](ShamrockCtx &ctx, std::string vector_type, std::string kernel) -> VariantGSPHModelBind {
            VariantGSPHModelBind ret;

            if (vector_type == "f64_3" && kernel == "M4") {
                ret = std::make_unique<Model<f64_3, shammath::M4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M6") {
                ret = std::make_unique<Model<f64_3, shammath::M6>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "M8") {
                ret = std::make_unique<Model<f64_3, shammath::M8>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C2") {
                ret = std::make_unique<Model<f64_3, shammath::C2>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C4") {
                ret = std::make_unique<Model<f64_3, shammath::C4>>(ctx);
            } else if (vector_type == "f64_3" && kernel == "C6") {
                ret = std::make_unique<Model<f64_3, shammath::C6>>(ctx);
            } else {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "unknown combination of representation and kernel");
            }

            return ret;
        },
        py::kw_only(),
        py::arg("context"),
        py::arg("vector_type"),
        py::arg("sph_kernel"),
        R"==(
    Create a GSPH (Godunov SPH) model.

    GSPH uses Riemann solvers at particle interfaces instead of artificial viscosity,
    giving sharper shock resolution.

    Parameters
    ----------
    context : ShamrockCtx
        Shamrock context
    vector_type : str
        Vector type, e.g., "f64_3" for 3D double precision
    sph_kernel : str
        SPH kernel type: "M4" (cubic spline), "M6", "M8", "C2", "C4", "C6" (Wendland)

    Returns
    -------
    GSPHModel
        A GSPH model instance

    Examples
    --------
    >>> ctx = shamrock.ShamrockCtx()
    >>> model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    >>> config = model.gen_default_config()
    >>> config.set_riemann_hllc()
    >>> config.set_eos_adiabatic(1.4)
    >>> model.set_solver_config(config)
)==");
}
