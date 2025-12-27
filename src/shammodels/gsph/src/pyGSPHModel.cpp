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
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
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
        // Reconstruction config
        .def(
            "set_reconstruct_first_order",
            [](TConfig &self) {
                self.set_reconstruct_first_order();
            },
            R"==(
    Set first-order piecewise constant reconstruction.

    Sets all gradients to zero. Most diffusive but most stable.
    Good for very strong shocks or initial testing.
)==")
        .def(
            "set_reconstruct_second_order",
            [](TConfig &self, Tscal mach_threshold) {
                self.set_reconstruct_second_order(mach_threshold);
            },
            py::arg("mach_threshold") = Tscal{1.1},
            R"==(
    Set second-order reconstruction (Inutsuka 2002).

    Uses computed gradients with monotonicity constraint.
    At shock fronts (Mach > mach_threshold), velocity gradients
    are set to zero to maintain stability.

    Parameters
    ----------
    mach_threshold : float, optional
        Mach number threshold for monotonicity constraint.
        When relative velocity > c_s / mach_threshold, use first-order.
        Default: 1.1
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
            "set_field_in_box",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 box_min,
               f64_3 box_max,
               u32 ivar) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_field_in_box(field_name, val, {box_min, box_max}, ivar);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_field_in_box(field_name, val, {box_min, box_max}, ivar);
                } else if (field_type == "u32") {
                    u32 val = value.cast<u32>();
                    self.set_field_in_box(field_name, val, {box_min, box_max}, ivar);
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

    Useful for setting up discontinuous initial conditions like Sod shock tube.

    Parameters
    ----------
    field_name : str
        Name of the field to set (e.g., "vxyz", "uint", "hpart")
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
    >>> # Sod shock tube: set left state internal energy
    >>> model.set_field_in_box("uint", "f64", u_left, (-1,-1,-1), (0,1,1))
    >>> # Set right state
    >>> model.set_field_in_box("uint", "f64", u_right, (0,-1,-1), (1,1,1))
)==")
        .def(
            "set_field_in_sphere",
            [](T &self,
               std::string field_name,
               std::string field_type,
               pybind11::object value,
               f64_3 center,
               f64 radius) {
                if (field_type == "f64") {
                    f64 val = value.cast<f64>();
                    self.set_field_in_sphere(field_name, val, center, radius);
                } else if (field_type == "f64_3") {
                    f64_3 val = value.cast<f64_3>();
                    self.set_field_in_sphere(field_name, val, center, radius);
                } else {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "unknown field type");
                }
            },
            py::arg("field_name"),
            py::arg("field_type"),
            py::arg("value"),
            py::arg("center"),
            py::arg("radius"),
            R"==(
    Set field value for particles within a spherical region.

    Useful for setting up point-source initial conditions like Sedov blast.

    Parameters
    ----------
    field_name : str
        Name of the field to set (e.g., "uint")
    field_type : str
        Type of the field: "f64" or "f64_3"
    value : float or tuple
        Value to set (type must match field_type)
    center : tuple
        Center of the sphere (x, y, z)
    radius : float
        Radius of the sphere

    Examples
    --------
    >>> # Sedov blast: inject energy in central sphere
    >>> model.set_field_in_sphere("uint", "f64", u_blast, (0,0,0), r_blast)
)==")
        .def("apply_field_from_position_f64_3", &T::template apply_field_from_position<f64_3>)
        .def("apply_field_from_position_f64", &T::template apply_field_from_position<f64>)
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
        .def(
            "load_from_dump",
            &T::load_from_dump,
            py::arg("filename"),
            R"==(
    Load simulation state from a Shamrock dump file.

    Uses the shared ShamrockDump mechanism (same as SPH).

    Parameters
    ----------
    filename : str
        Path to the dump file

    Example
    -------
    >>> model.load_from_dump("checkpoint.shamrock")
)==")
        .def(
            "dump",
            &T::dump,
            py::arg("filename"),
            R"==(
    Write simulation state to a Shamrock dump file.

    Uses the shared ShamrockDump mechanism (same as SPH).

    Parameters
    ----------
    filename : str
        Path to the dump file

    Example
    -------
    >>> model.dump("checkpoint.shamrock")
)==");
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
        py::arg("vector_type") = "f64_3",
        py::arg("sph_kernel")  = "C4",
        R"==(
    Create a GSPH (Godunov SPH) model.

    GSPH uses Riemann solvers at particle interfaces instead of artificial viscosity,
    giving sharper shock resolution.

    Parameters
    ----------
    context : ShamrockCtx
        Shamrock context
    vector_type : str
        Vector type, e.g., "f64_3" for 3D double precision (default: "f64_3")
    sph_kernel : str
        SPH kernel type: "C4" (Wendland, default), "M4" (cubic spline), "M6", "M8", "C2", "C6"

    Returns
    -------
    GSPHModel
        A GSPH model instance

    Examples
    --------
    >>> ctx = shamrock.ShamrockCtx()
    >>> model = shamrock.get_Model_GSPH(context=ctx)  # Uses C4 kernel by default
    >>> config = model.gen_default_config()
    >>> config.set_riemann_hllc()
    >>> config.set_eos_adiabatic(1.4)
    >>> model.set_solver_config(config)
)==");
}
