// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyGSPHModel.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
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
#include "shammodels/common/shamrock_json_to_py_json.hpp"
#include "shammodels/gsph/Model.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <memory>

template<class Tvec, template<class> class SPHKernel>
void add_gsph_instance(py::module &m, std::string name_config, std::string name_model) {
    using namespace shammodels::gsph;

    using Tscal = shambase::VecComponent<Tvec>;

    using T          = Model<Tvec, SPHKernel>;
    using TGSPHSetup = modules::GSPHSetup<Tvec, SPHKernel>;
    using TConfig    = typename T::SolverConfig;

    shamlog_debug_ln("[Py]", "registering class :", name_config, typeid(T).name());
    shamlog_debug_ln("[Py]", "registering class :", name_model, typeid(T).name());

    py::class_<TConfig> config_cls(m, name_config.c_str());

    shammodels::common::add_json_defs<TConfig>(config_cls);

    config_cls.def("print_status", &TConfig::print_status)
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
            [](TConfig &self, Tscal tol, u32 max_iter) {
                self.set_riemann_exact(tol, max_iter);
            },
            py::kw_only(),
            py::arg("tolerance") = Tscal{1e-8},
            py::arg("max_iter")  = 100,
            R"==(
    Set exact Riemann solver (Toro 2009).

    Classifies the wave pattern (shock/rarefaction on each side) from the
    initial states, then solves the matching closed-form relation via
    bisection. Most accurate but computationally expensive; unlike the
    iterative (van Leer) solver, it also remains accurate for strong
    rarefactions / near-vacuum conditions.

    Parameters
    ----------
    tolerance : float
        Bisection convergence tolerance (default: 1e-8)
    max_iter : int
        Maximum number of bisection iterations (default: 100)
)==")
        // Reconstruction config
        .def(
            "set_reconstruct_piecewise_constant",
            [](TConfig &self) {
                self.set_reconstruct_piecewise_constant();
            },
            R"==(
    Set first-order piecewise constant reconstruction.

    Sets all gradients to zero. Most diffusive but most stable.
    Good for very strong shocks or initial testing.
)==")
        // Force formulation config
        .def(
            "set_force_cha_whitworth",
            [](TConfig &self) {
                self.set_force_cha_whitworth();
            },
            R"==(
    Set the Cha & Whitworth (2003) symmetric SPH force formulation (default).

    Uses the standard SPH momentum equation (nabla_W/rho^2/Omega) with the
    Riemann-solved interface pressure p* substituted for pressure.
)==")
        .def(
            "set_force_inutsuka_v2",
            [](TConfig &self) {
                self.set_force_inutsuka_v2();
            },
            R"==(
    Set the Inutsuka (2002) effective volume/face force formulation.

    Uses linear (1st order) interpolation of the volume element between each
    particle pair to build an effective face (V2_ij, s*), following the
    original GSPH momentum equation: acc -= m * p* * V2_ij * grad_W_ij.
)==")
        // EOS config
        .def(
            "set_eos_adiabatic",
            [](TConfig &self, Tscal gamma) {
                self.set_eos_adiabatic(gamma);
            },
            py::arg("gamma"),
            R"==(
    Set adiabatic equation of state: P = (\gamma-1)  \rho  u

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
    Set isothermal equation of state: P = cs^2  \rho

    Parameters
    ----------
    cs : float
        Sound speed
)==")
        .def(
            "set_eos_locally_isothermalFA2014",
            [](TConfig &self, Tscal h_over_r) {
                self.set_eos_locally_isothermalFA2014(h_over_r);
            },
            py::kw_only(),
            py::arg("h_over_r"))
        .def(
            "set_eos_locally_isothermalFA2014_extended",
            [](TConfig &self, Tscal cs0, Tscal q, Tscal r0, u32 n_sinks) {
                self.set_eos_locally_isothermalFA2014_extended(cs0, q, r0, n_sinks);
            },
            py::kw_only(),
            py::arg("cs0"),
            py::arg("q"),
            py::arg("r0"),
            py::arg("n_sinks"))
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
        .def(
            "set_scheduler_config",
            [](TConfig &self, u64 split_crit, u64 merge_crit) {
                self.scheduler_conf.split_load_value = split_crit;
                self.scheduler_conf.merge_load_value = merge_crit;
            },
            py::kw_only(),
            py::arg("split_load_value"),
            py::arg("merge_load_value"));

    std::string setup_name = name_model + "_GSPHSetup";
    py::class_<TGSPHSetup>(m, setup_name.c_str())
        .def(
            "make_generator_disc_mc",
            [](TGSPHSetup &self,
               Tscal part_mass,
               Tscal disc_mass,
               Tscal r_in,
               Tscal r_out,
               std::function<Tscal(Tscal)> sigma_profile,
               std::function<Tscal(Tscal)> H_profile,
               std::function<Tscal(Tscal)> rot_profile,
               std::function<Tscal(Tscal)> cs_profile,
               std::function<Tvec(Tvec)> velocity_field,
               std::function<Tscal(Tvec)> cs_field,
               u64 random_seed,
               Tscal init_h_factor) {
                auto build_vel_lambda = [&]() -> std::function<Tvec(Tvec)> {
                    if (!velocity_field && !rot_profile) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "make_generator_disc_mc: either velocity_field or rot_profile must be "
                            "provided, you must provide one of them");
                    }

                    if (velocity_field && rot_profile) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "make_generator_disc_mc: either velocity_field or rot_profile must be "
                            "provided, you cannot provide both");
                    }

                    if (velocity_field) {
                        return std::move(velocity_field);
                    }
                    return [vth_r = std::move(rot_profile)](Tvec pos) {
                        pos[2]  = 0; // to get the cylindrical radius
                        Tscal r = sycl::length(pos);

                        auto etheta = sycl::vec<Tscal, 3>{-pos.y(), pos.x(), 0};
                        etheta /= sycl::length(etheta);

                        return vth_r(r) * etheta;
                    };
                };

                auto build_cs_lambda = [&]() -> std::function<Tscal(Tvec)> {
                    bool need_cs = false; // self.solver_config.is_eos_locally_isothermal();

                    if (!need_cs) {
                        if (cs_field) {
                            if (shamcomm::world_rank() == 0) {
                                logger::warn_ln(
                                    "GSPHSetup",
                                    "make_generator_disc_mc: with the current EOS, cs_field is "
                                    "ignored");
                            }
                        }
                        if (cs_profile) {
                            if (shamcomm::world_rank() == 0) {
                                logger::warn_ln(
                                    "GSPHSetup",
                                    "make_generator_disc_mc: with the current EOS, cs_profile is "
                                    "ignored");
                            }
                        }
                        return std::function<Tscal(Tvec)>{};
                    }

                    if (!cs_field && !cs_profile) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "make_generator_disc_mc: either cs_field or cs_profile must be "
                            "provided, you must provide one of them");
                    }

                    if (cs_field && cs_profile) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "make_generator_disc_mc: either cs_field or cs_profile must be "
                            "provided, you cannot provide both");
                    }

                    if (cs_field) {
                        return std::move(cs_field);
                    }

                    return [cs_r = std::move(cs_profile)](Tvec pos) {
                        pos[2]  = 0; // to get the cylindrical radius
                        Tscal r = sycl::length(pos);
                        return cs_r(r);
                    };
                };

                return self.make_generator_disc_mc(
                    part_mass,
                    disc_mass,
                    r_in,
                    r_out,
                    std::move(sigma_profile),
                    std::move(H_profile),
                    build_vel_lambda(),
                    build_cs_lambda(),
                    std::mt19937_64(random_seed),
                    init_h_factor);
            },
            py::kw_only(),
            py::arg("part_mass"),
            py::arg("disc_mass"),
            py::arg("r_in"),
            py::arg("r_out"),
            py::arg("sigma_profile"),
            py::arg("H_profile"),
            py::arg("rot_profile")    = std::function<Tscal(Tscal)>{},
            py::arg("cs_profile")     = std::function<Tscal(Tscal)>{},
            py::arg("velocity_field") = std::function<Tvec(Tvec)>{},
            py::arg("cs_field")       = std::function<Tscal(Tvec)>{},
            py::arg("random_seed"),
            py::arg("init_h_factor") = 0.8,
            R"pbdoc(
        Create a Monte Carlo disc particle generator.

        Particles are sampled in cylindrical coordinates: the radius is drawn
        with rejection sampling from ``sigma_profile``, the azimuth is uniform,
        and the vertical coordinate follows a Gaussian with scale ``H_profile(r)``.
        The initial density is extrapolated from the surface density profile, and
        smoothing lengths are set from that density.

        Args:
            part_mass: Mass of each GSPH particle.
            disc_mass: Total disc mass. The particle count is ``disc_mass / part_mass``.
            r_in: Inner disc radius.
            r_out: Outer disc radius.
            sigma_profile: Surface density profile ``sigma(r)``.
            H_profile: Disc scale height profile ``H(r)``.
            rot_profile: Azimuthal speed profile ``v_theta(r)``. The velocity is
                projected along the cylindrical azimuthal direction at each
                particle position. Mutually exclusive with ``velocity_field``.
            cs_profile: Sound speed profile ``c_s(r)``. Evaluated at the cylindrical
                radius of each particle. Required when the solver uses a locally
                isothermal EOS. Mutually exclusive with ``cs_field``.
            velocity_field: Velocity profile ``v(x, y, z)``. Mutually exclusive
                with ``rot_profile``.
            cs_field: Sound speed profile ``c_s(x, y, z)``. Required when the solver
                uses a locally isothermal EOS. Mutually exclusive with ``cs_profile``.
            random_seed: Seed for the Monte Carlo sampler.
            init_h_factor: Multiplier applied to the smoothing length inferred from
                the generated density. Defaults to ``0.8``.

        Notes:
            Exactly one of ``velocity_field`` or ``rot_profile`` must be provided.

            If the solver uses a locally isothermal EOS, exactly one of ``cs_field``
            or ``cs_profile`` must be provided. Otherwise both sound-speed profiles
            are ignored and a warning is emitted if either is supplied.

        Returns:
            A setup node to pass to :py:meth:`apply_setup`.
    )pbdoc")
        .def(
            "apply_setup",
            [](TGSPHSetup &self,
               modules::SetupNodePtr setup,
               std::optional<u32> gen_step,
               std::optional<u32> insert_step,
               std::optional<u64> msg_count_limit,
               std::optional<u64> msg_size_limit,
               std::optional<u64> max_msg_size,
               bool do_setup_log,
               bool use_new_setup,
               bool speculative_balancing) {
                if (bool(gen_step)) {
                    ON_RANK_0(
                        logger::warn_ln("GSPHSetup", "gen_step is ignored when using old setup"));
                }
                if (bool(msg_count_limit)) {
                    ON_RANK_0(
                        logger::warn_ln(
                            "GSPHSetup", "msg_count_limit is ignored when using old setup"));
                }
                if (bool(msg_size_limit)) {
                    ON_RANK_0(
                        logger::warn_ln(
                            "GSPHSetup", "msg_size_limit is ignored when using old setup"));
                }
                if (bool(max_msg_size)) {
                    ON_RANK_0(
                        logger::warn_ln(
                            "GSPHSetup", "max_msg_size is ignored when using old setup"));
                }
                if (bool(do_setup_log)) {
                    ON_RANK_0(
                        logger::warn_ln(
                            "GSPHSetup", "do_setup_log is ignored when using old setup"));
                }
                return self.apply_setup(setup, insert_step);
            },
            py::arg("setup"),
            py::kw_only(),
            py::arg("gen_step")              = std::nullopt,
            py::arg("insert_step")           = std::nullopt,
            py::arg("msg_count_limit")       = std::nullopt,
            py::arg("rank_comm_size_limit")  = std::nullopt,
            py::arg("max_msg_size")          = std::nullopt,
            py::arg("do_setup_log")          = false,
            py::arg("use_new_setup")         = true,
            py::arg("speculative_balancing") = false);

    py::class_<T>(m, name_model.c_str())
        .def(py::init([](ShamrockCtx &ctx) {
            return std::make_unique<T>(ctx);
        }))
        .def("init", &T::init)
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
        .def("add_sink", &T::add_sink)
        .def(
            "get_sinks",
            [](T &self) {
                py::list list_out;

                if (!self.solver.storage.sinks.is_empty()) {
                    for (auto &sink : self.solver.storage.sinks.get()) {
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
                }

                return list_out;
            })
        .def("do_vtk_dump", &T::do_vtk_dump)
        .def("solver_logs_last_rate", &T::solver_logs_last_rate)
        .def("solver_logs_last_obj_count", &T::solver_logs_last_obj_count)
        .def(
            "get_time",
            [](T &self) {
                return self.solver.get_time();
            })
        .def(
            "get_dt",
            [](T &self) {
                return self.solver.get_dt();
            })
        .def(
            "set_time",
            [](T &self, Tscal t) {
                return self.solver.set_time(t);
            })
        .def(
            "set_next_dt",
            [](T &self, Tscal dt) {
                return self.solver.set_next_dt(dt);
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
)==")
        .def("get_setup", &T::get_setup);
}

using namespace shammodels::gsph;

ON_PYTHON_INIT {
    auto &m = root_module;

    py::module mgsph = m.def_submodule("model_gsph", "Shamrock GSPH (Godunov SPH) solver");

    using namespace shammodels::gsph;

    py::class_<
        shammodels::gsph::modules::IGSPHSetupNode,
        std::shared_ptr<shammodels::gsph::modules::IGSPHSetupNode>>(mgsph, "IGSPHSetupNode")
        .def("get_dot", [](std::shared_ptr<shammodels::gsph::modules::IGSPHSetupNode> &self) {
            return self->get_dot();
        });

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
        py::arg("sph_kernel")  = "M4",
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
        SPH kernel type: "M4" (cubic spline, default), "M6", "M8" (quintic spline),
        "C2", "C4", "C6" (Wendland kernels)

    Returns
    -------
    GSPHModel
        A GSPH model instance

    Examples
    --------
    >>> ctx = shamrock.ShamrockCtx()
    >>> model = shamrock.get_Model_GSPH(context=ctx)  # Uses M4 kernel by default
    >>> config = model.gen_default_config()
    >>> config.set_riemann_hllc()
    >>> config.set_eos_adiabatic(1.4)
    >>> model.set_solver_config(config)
)==");
}
