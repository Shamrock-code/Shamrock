// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySolverConfig.cpp
 * @author David Fang (david.fang@ikmail.com) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Python bindings for shammodels::sph::SolverConfig
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/common/config/enum_TensileCorrection.hpp"
#include "shammodels/common/shamrock_json_to_py_json.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "sph_pybindings.hpp"
#include <experimental/mdspan>
#include <pybind11/numpy.h>
#include <optional>

namespace shammodels::sph::pybindings {

    template<class Tvec, template<class> class SPHKernel>
    void add_solver_config_instance(py::module &m, const std::string &name_config) {
        using namespace shammodels::sph;

        using Tscal = shambase::VecComponent<Tvec>;

        using TConfig = SolverConfig<Tvec, SPHKernel>;

        py::class_<TConfig> config_cls(m, name_config.c_str());

        shammodels::common::add_json_defs<TConfig>(config_cls);

        config_cls.def("print_status", &TConfig::print_status);
        config_cls.def("set_particle_tracking", &TConfig::set_particle_tracking);
        config_cls.def(
            "set_scheduler_config",
            [](TConfig &self, u64 split_crit, u64 merge_crit) {
                self.scheduler_conf.split_load_value = split_crit;
                self.scheduler_conf.merge_load_value = merge_crit;
            },
            py::kw_only(),
            py::arg("split_load_value"),
            py::arg("merge_load_value"));
        config_cls.def("set_tree_reduction_level", &TConfig::set_tree_reduction_level);
        config_cls.def(
            "set_neigh_cache_strategy",
            &TConfig::set_neigh_cache_strategy,
            R"==(
    Set the strategy used to build the neighbours cache.

    Parameters
    ----------
    strategy : NeighCacheStrategy
        Either ``NeighCacheStrategy.SingleStage`` or ``NeighCacheStrategy.TwoStage``
        (the default), as obtained from ``from shamrock import NeighCacheStrategy``.
)==");
        config_cls.def(
            "set_two_stage_search",
            &TConfig::set_two_stage_search,
            R"==(
    Set the neighbours cache strategy from a boolean.

    .. deprecated::
        Use :py:meth:`set_neigh_cache_strategy` instead.
)==");
        config_cls.def("set_show_neigh_stats", &TConfig::set_show_neigh_stats);
        config_cls.def(
            "set_max_neigh_cache_size", [](TConfig &self, const py::object &max_neigh_cache_size) {
                ON_RANK_0(shamlog_warn_ln(
                              "SPH",
                              ".set_max_neigh_cache_size() is deprecated,\n"
                              "    -> calling this is a no-op,\n"
                              "    -> you can remove the call to that function"););
            });
        config_cls.def("set_filter_empty_patch_gz", &TConfig::set_filter_empty_patch_gz);
        config_cls.def(
            "set_smoothing_length_density_based", &TConfig::set_smoothing_length_density_based);
        config_cls.def(
            "set_smoothing_length_density_based_neigh_lim",
            &TConfig::set_smoothing_length_density_based_neigh_lim);
        config_cls.def(
            "set_smoothing_length_density_based_bisecting_nr",
            &TConfig::set_smoothing_length_density_based_bisecting_nr);
        config_cls.def("set_enable_particle_reordering", &TConfig::set_enable_particle_reordering);
        config_cls.def(
            "set_particle_reordering_step_freq", &TConfig::set_particle_reordering_step_freq);
        config_cls.def("set_show_ghost_zone_graph", &TConfig::set_show_ghost_zone_graph);
        config_cls.def("set_show_ghost_stats", &TConfig::set_show_ghost_stats);
        config_cls.def("use_luminosity", &TConfig::use_luminosity);
        config_cls.def("use_div_b", &TConfig::use_div_b);
        config_cls.def("compute_GW", &TConfig::use_GW);
        config_cls.def("set_save_dt_to_fields", &TConfig::set_save_dt_to_fields);
        config_cls.def("should_save_dt_to_fields", &TConfig::should_save_dt_to_fields);
        config_cls.def("set_eos_isothermal", &TConfig::set_eos_isothermal);
        config_cls.def("set_eos_adiabatic", &TConfig::set_eos_adiabatic);
        config_cls.def("set_eos_polytropic", &TConfig::set_eos_polytropic);
        config_cls.def("set_eos_locally_isothermal", &TConfig::set_eos_locally_isothermal);
        config_cls.def(
            "set_eos_locally_isothermalLP07",
            [](TConfig &self, Tscal cs0, Tscal q, Tscal r0) {
                self.set_eos_locally_isothermalLP07(cs0, q, r0);
            },
            py::kw_only(),
            py::arg("cs0"),
            py::arg("q"),
            py::arg("r0"));
        config_cls.def(
            "set_eos_locally_isothermalFA2014",
            [](TConfig &self, Tscal h_over_r) {
                self.set_eos_locally_isothermalFA2014(h_over_r);
            },
            py::kw_only(),
            py::arg("h_over_r"));
        config_cls.def(
            "set_eos_locally_isothermalFA2014_extended",
            [](TConfig &self, Tscal cs0, Tscal q, Tscal r0, u32 n_sinks) {
                self.set_eos_locally_isothermalFA2014_extended(cs0, q, r0, n_sinks);
            },
            py::kw_only(),
            py::arg("cs0"),
            py::arg("q"),
            py::arg("r0"),
            py::arg("n_sinks"));
        config_cls.def(
            "set_eos_fermi",
            [](TConfig &self, Tscal mu_e) {
                self.set_eos_fermi(mu_e);
            },
            py::kw_only(),
            py::arg("mu_e"));
        config_cls.def(
            "set_eos_barotropic",
            [](TConfig &self, Tscal cs0, Tscal rho0, Tscal gamma0) {
                self.set_eos_barotropic(cs0, rho0, gamma0);
            },
            py::kw_only(),
            py::arg("cs0"),
            py::arg("rho0"),
            py::arg("gamma0") = Tscal{5.0 / 3.0});
        config_cls.def("set_artif_viscosity_None", &TConfig::set_artif_viscosity_None);
        config_cls.def(
            "set_artif_viscosity_Constant",
            [](TConfig &self, Tscal alpha_u, Tscal alpha_AV, Tscal beta_AV) {
                self.set_artif_viscosity_Constant({alpha_u, alpha_AV, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_u"),
            py::arg("alpha_AV"),
            py::arg("beta_AV"));
        config_cls.def(
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
            py::arg("beta_AV"));
        config_cls.def(
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
            py::arg("beta_AV"));
        config_cls.def(
            "set_artif_viscosity_ConstantDisc",
            [](TConfig &self, Tscal alpha_AV, Tscal alpha_u, Tscal beta_AV) {
                self.set_artif_viscosity_ConstantDisc({alpha_AV, alpha_u, beta_AV});
            },
            py::kw_only(),
            py::arg("alpha_AV"),
            py::arg("alpha_u"),
            py::arg("beta_AV"));
        config_cls.def(
            "set_IdealMHD",
            [](TConfig &self,
               Tscal sigma_mhd,
               Tscal alpha_B,
               shammodels::TensileCorrection tensile_corr) {
                self.set_ideal_mhd({sigma_mhd, alpha_B, tensile_corr});
            },
            py::kw_only(),
            py::arg("sigma_mhd"),
            py::arg("alpha_B")      = 1.0,
            py::arg("tensile_corr") = TensileCorrection::Borve,
            R"==(
    Enable ideal SPMHD.

    Parameters
    ----------
    sigma_mhd : float
    alpha_B : float, default 1.0
    tensile_corr : TensileCorrection, default TensileCorrection.Borve
        Tensile instability correction.
)==");
        config_cls.def(
            "set_NonIdealMHD",
            [](TConfig &self,
               Tscal sigma_mhd,
               Tscal etaO,
               Tscal etaH,
               Tscal etaAD,
               Tscal alpha_B,
               shammodels::TensileCorrection tensile_corr) {
                self.set_non_ideal_mhd({sigma_mhd, alpha_B, etaO, etaH, etaAD, tensile_corr});
            },
            py::kw_only(),
            py::arg("sigma_mhd"),
            py::arg("etaO"),
            py::arg("etaH"),
            py::arg("etaAD"),
            py::arg("alpha_B")      = 1.0,
            py::arg("tensile_corr") = TensileCorrection::Borve,
            R"==(
    Enable non-ideal SPMHD.

    Parameters
    ----------
    sigma_mhd : float
    etaO, etaH, etaAD : float
        Ohmic, Hall and ambipolar diffusion coefficients.
    alpha_B : float, default 1.0
    tensile_corr : TensileCorrection, default TensileCorrection.Borve
        Tensile instability correction.
)==");
        config_cls.def("set_self_gravity_none", [](TConfig &self) {
            self.self_grav_config.set_none();
        });
        config_cls.def(
            "set_self_gravity_direct",
            [](TConfig &self, bool reference_mode = false) {
                self.self_grav_config.set_direct(reference_mode);
            },
            py::kw_only(),
            py::arg("reference_mode") = false);
        config_cls.def(
            "set_self_gravity_mm",
            [](TConfig &self, u32 mm_order, f64 opening_angle, u32 reduction_level) {
                self.self_grav_config.set_mm(mm_order, opening_angle, reduction_level);
            },
            py::kw_only(),
            py::arg("order"),
            py::arg("opening_angle"),
            py::arg("reduction_level") = 3);
        config_cls.def(
            "set_self_gravity_fmm",
            [](TConfig &self, u32 order, f64 opening_angle, u32 reduction_level) {
                self.self_grav_config.set_fmm(order, opening_angle, reduction_level);
            },
            py::kw_only(),
            py::arg("order"),
            py::arg("opening_angle"),
            py::arg("reduction_level") = 3);
        config_cls.def(
            "set_self_gravity_sfmm",
            [](TConfig &self,
               u32 sfmm_order,
               f64 opening_angle,
               bool leaf_lowering,
               u32 reduction_level) {
                self.self_grav_config.set_sfmm(
                    sfmm_order, opening_angle, leaf_lowering, reduction_level);
            },
            py::kw_only(),
            py::arg("order"),
            py::arg("opening_angle"),
            py::arg("leaf_lowering")   = true,
            py::arg("reduction_level") = 3);
        config_cls.def(
            "set_softening_plummer",
            [](TConfig &self, f64 epsilon) {
                self.self_grav_config.set_softening_plummer(epsilon);
            },
            py::kw_only(),
            py::arg("epsilon"));
        config_cls.def("set_softening_none", [](TConfig &self) {
            self.self_grav_config.set_softening_none();
        });
        config_cls.def("set_boundary_free", &TConfig::set_boundary_free);
        config_cls.def("set_boundary_periodic", &TConfig::set_boundary_periodic);
        config_cls.def("set_boundary_shearing_periodic", &TConfig::set_boundary_shearing_periodic);
        config_cls.def("set_dust_mode_none", [](TConfig &self) {
            self.dust_config.set_none();
        });
        config_cls.def(
            "set_dust_mode_monofluid_tva",
            [](TConfig &self,
               u32 nvar,
               bool pure_diffusion_mode,
               Tscal C_1_fluid,
               Tscal C_drift,
               Tscal cfl_density_threshold,
               bool ensure_s_j_positivity,
               bool smooth_s_positivity_limiter,
               bool dust_corrected_av,
               std::optional<Tscal> clamp_dust_frac,
               bool pair_gate_s_positivity_limiter) {
                self.dust_config.set_monofluid_tva(
                    nvar,
                    pure_diffusion_mode,
                    C_1_fluid,
                    C_drift,
                    cfl_density_threshold,
                    ensure_s_j_positivity,
                    smooth_s_positivity_limiter,
                    dust_corrected_av,
                    clamp_dust_frac,
                    pair_gate_s_positivity_limiter);
            },
            py::kw_only(),
            py::arg("nvar"),
            py::arg("pure_diffusion_mode")            = false,
            py::arg("C_1_fluid")                      = 0.1,
            py::arg("C_drift")                        = 1.0,
            py::arg("cfl_density_threshold")          = shambase::get_epsilon<Tscal>(),
            py::arg("ensure_s_j_positivity")          = true,
            py::arg("smooth_s_positivity_limiter")    = false,
            py::arg("dust_corrected_av")              = false,
            py::arg("clamp_dust_frac")                = std::nullopt,
            py::arg("pair_gate_s_positivity_limiter") = false);
        config_cls.def(
            "set_dust_mode_monofluid_complete",
            [](TConfig &self, u32 ndust) {
                self.dust_config.set_monofluid_complete(ndust);
            },
            py::kw_only(),
            py::arg("ndust"));
        config_cls.def("set_dust_drag_constant", [](TConfig &self, std::vector<Tscal> ts) {
            self.dust_config.set_drag_constant({.stopping_times = std::move(ts)});
        });
        config_cls.def(
            "set_dust_drag_epstein",
            [](TConfig &self,
               Tscal gamma,
               std::vector<Tscal> grain_sizes,
               std::vector<Tscal> grain_densities) {
                self.dust_config.set_drag_epstein(
                    {.gamma            = gamma,
                     .grains_sizes     = std::move(grain_sizes),
                     .grains_densities = std::move(grain_densities)});
            },
            py::arg("gamma"),
            py::arg("grain_sizes"),
            py::arg("grain_densities"));
        config_cls.def(
            "set_dust_evol_coala_coag",
            [](TConfig &self,
               Tscal rhodust_eps,
               Tscal vfrag_threshold,
               std::vector<Tscal> massgrid,
               py::array_t<Tscal> tabflux_coag,
               Tscal tau_frac) {
                if (massgrid.size() == 0) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "massgrid must not be empty");
                }

                u32 nbins = massgrid.size() - 1;

                // tabflux_coag is a 3D array of shape (nbins ** 3)

                // assert rank is 3
                if (tabflux_coag.ndim() != 3) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "tabflux_coag must be a 3D array, got ndim="
                        + std::to_string(tabflux_coag.ndim()));
                }

                // assert shape is (nbins, nbins, nbins)
                if (tabflux_coag.shape(0) != nbins || tabflux_coag.shape(1) != nbins
                    || tabflux_coag.shape(2) != nbins) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(
                        "tabflux_coag must be a 3D array of shape (nbins, nbins, nbins) with "
                        "nbins="
                        + std::to_string(nbins) + " (massgrid.size() - 1), got shape ("
                        + std::to_string(tabflux_coag.shape(0)) + ", "
                        + std::to_string(tabflux_coag.shape(1)) + ", "
                        + std::to_string(tabflux_coag.shape(2)) + ")");
                }

                std::vector<Tscal> tabflux_coag_vec(nbins * nbins * nbins);

                using mdspan_rank_3 = std::mdspan<Tscal, std::dextents<u32, 3>>;
                mdspan_rank_3 tabflux_coag_mdspan(tabflux_coag_vec.data(), nbins, nbins, nbins);

                for (u32 i = 0; i < nbins; i++) {
                    for (u32 j = 0; j < nbins; j++) {
                        for (u32 k = 0; k < nbins; k++) {
                            tabflux_coag_mdspan(i, j, k) = tabflux_coag.mutable_at(i, j, k);
                        }
                    }
                }

                self.dust_config.set_dust_evol_coala(
                    {.rhodust_eps     = rhodust_eps,
                     .tau_frac        = tau_frac,
                     .vfrag_threshold = vfrag_threshold,
                     .massgrid        = massgrid,
                     .tabflux_coag    = tabflux_coag_vec});
            },
            py::arg("rhodust_eps"),
            py::arg("vfrag_threshold"),
            py::arg("massgrid"),
            py::arg("tabflux_coag"),
            py::arg("tau_frac") = Tscal(10),
            R"pbdoc(
        Enable the COALA dust coagulation source term.

        Args:
            rhodust_eps: Dust density floor.
            vfrag_threshold: Fragmentation velocity threshold, must be positive.
                In this coagulation-only mode, a pair of dust bins whose
                differential velocity ``|dv_ij|`` exceeds this value gets
                ``dv_ij = 0``. This is a crude stand-in for fragmentation
                ("poor man" fragmentation), not a fragmentation model. Pass
                ``math.inf`` to disable it.
            massgrid: Dust mass grid bin edges, of size ``nbins + 1``.
            tabflux_coag: Tabulated coagulation flux, of shape
                ``(nbins, nbins, nbins)``.
            tau_frac: The source term on ``s_j`` is integrated implicitly over
                ``dt_eff = max(dt, tau_frac * rhodust_eps / |S|)``, that is over at
                least the time needed to move ``tau_frac * rhodust_eps``. This bounds
                the rate of an empty bin when ``dt`` is null or tiny (e.g. right after
                a dust injection). Must be positive.
        )pbdoc");
        config_cls.def(
            "set_dust_ballabio_ts_limiter",
            [](TConfig &self, bool enabled) {
                self.dust_config.ballabio_ts_limiter = enabled;
            },
            py::arg("enabled"));
        config_cls.def(
            "add_ext_force_point_mass",
            [](TConfig &self, Tscal central_mass, Tscal Racc, Tvec central_pos) {
                self.add_ext_force_point_mass(central_mass, Racc, central_pos);
            },
            py::arg("central_mass"),
            py::arg("Racc"),
            py::kw_only(),
            py::arg("central_pos") = Tvec{0, 0, 0});
        config_cls.def("add_ext_force_paczynski_wiita", &TConfig::add_ext_force_paczynski_wiita);
        config_cls.def(
            "add_ext_force_lense_thirring",
            [](TConfig &self,
               Tscal central_mass,
               Tscal Racc,
               Tscal a_spin,
               Tvec dir_spin,
               Tvec central_pos) {
                self.add_ext_force_lense_thirring(
                    central_mass, Racc, a_spin, dir_spin, central_pos);
            },
            py::kw_only(),
            py::arg("central_mass"),
            py::arg("Racc"),
            py::arg("a_spin"),
            py::arg("dir_spin"),
            py::arg("central_pos") = Tvec{0, 0, 0});
        config_cls.def(
            "add_ext_force_shearing_box",
            [](TConfig &self, Tscal Omega_0, Tscal eta, Tscal q) {
                self.add_ext_force_shearing_box(Omega_0, eta, q);
            },
            py::kw_only(),
            py::arg("Omega_0"),
            py::arg("eta"),
            py::arg("q"));
        config_cls.def(
            "add_ext_force_velocity_dissipation",
            [](TConfig &self, Tscal eta) {
                self.ext_force_config.add_velocity_dissipation(eta);
            },
            py::kw_only(),
            py::arg("eta"));
        config_cls.def(
            "add_ext_force_vertical_disc_potential",
            [](TConfig &self, Tscal central_mass, Tscal R0) {
                self.ext_force_config.add_vertical_disc_potential(central_mass, R0);
            },
            py::kw_only(),
            py::arg("central_mass"),
            py::arg("R0"));
        config_cls.def("set_units", &TConfig::set_units);
        config_cls.def("get_units", [](TConfig &self) {
            return self.unit_sys;
        });
        config_cls.def("set_cfl_cour", [](TConfig &self, Tscal cfl_cour) {
            self.cfl_config.cfl_cour = cfl_cour;
        });
        config_cls.def("set_cfl_force", [](TConfig &self, Tscal cfl_force) {
            self.cfl_config.cfl_force = cfl_force;
        });
        config_cls.def("set_eta_sink", [](TConfig &self, Tscal eta_sink) {
            self.cfl_config.eta_sink = eta_sink;
        });
        config_cls.def("set_cfl_mult_stiffness", &TConfig::set_cfl_mult_stiffness);
        config_cls.def(
            "set_show_cfl_detail",
            [](TConfig &self, bool show_cfl_detail) {
                self.show_cfl_detail = show_cfl_detail;
            },
            py::arg("show_cfl_detail"));
        config_cls.def("set_particle_mass", [](TConfig &self, Tscal gpart_mass) {
            self.gpart_mass = gpart_mass;
        });
        config_cls.def(
            "add_kill_sphere",
            [](TConfig &self, const Tvec &center, Tscal radius) {
                self.particle_killing.add_kill_sphere(center, radius);
            },
            py::kw_only(),
            py::arg("center"),
            py::arg("radius"));
    }

    template void add_solver_config_instance<f64_3, shammath::M4>(
        py::module &, const std::string &);
    template void add_solver_config_instance<f64_3, shammath::M6>(
        py::module &, const std::string &);
    template void add_solver_config_instance<f64_3, shammath::M8>(
        py::module &, const std::string &);
    template void add_solver_config_instance<f64_3, shammath::C2>(
        py::module &, const std::string &);
    template void add_solver_config_instance<f64_3, shammath::C4>(
        py::module &, const std::string &);
    template void add_solver_config_instance<f64_3, shammath::C6>(
        py::module &, const std::string &);

} // namespace shammodels::sph::pybindings
