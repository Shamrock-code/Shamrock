// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Configuration for the Godunov SPH (GSPH) solver
 *
 * This file contains the main configuration structure for the GSPH solver.
 * GSPH uses Riemann solvers at particle interfaces instead of artificial viscosity.
 *
 * References:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of Godunov-type
 *   particle hydrodynamics"
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics with
 *   Riemann Solver"
 */

#include "shambase/exception.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/EOSConfig.hpp"
#include "shammodels/common/ExtForceConfig.hpp"
#include "shammodels/gsph/config/ForceFormulationConfig.hpp"
#include "shammodels/gsph/config/ReconstructConfig.hpp"
#include "shammodels/gsph/config/RiemannConfig.hpp"
#include "shammodels/sph/config/BCConfig.hpp" // Reuse boundary conditions from SPH
#include "shamrock/io/json_std_optional.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <nlohmann/json.hpp>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <variant>
#include <vector>

namespace shammodels::gsph {

    /**
     * @brief The configuration for a GSPH solver
     *
     * @tparam Tvec the type of the vector used to represent the particles
     * @tparam SPHKernel the type of the SPH kernel
     */
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    /**
     * @brief The configuration for the CFL condition in GSPH
     *
     * @tparam Tscal the type of the scalar used to represent the quantities
     */
    template<class Tscal>
    struct CFLConfig {
        Tscal cfl_cour  = 0.3;  ///< CFL condition for the courant factor
        Tscal cfl_force = 0.25; ///< CFL condition for the force
    };

    struct SmoothingLengthConfig {
        struct DensityBased {};
        struct DensityBasedNeighLim {
            u32 max_neigh_count = 500;
        };

        using mode = std::variant<DensityBased, DensityBasedNeighLim>;

        mode config = DensityBased{};

        void set_density_based() { config = DensityBased{}; }
        void set_density_based_neigh_lim(u32 max_neigh_count) {
            config = DensityBasedNeighLim{max_neigh_count};
        }

        bool is_density_based_neigh_lim() const {
            return std::holds_alternative<DensityBasedNeighLim>(config);
        }
    };

} // namespace shammodels::gsph

template<class Tvec, template<class> class SPHKernel>
struct shammodels::gsph::SolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass{0}; ///< The mass of each gas particle (must be set before use)

    CFLConfig<Tscal> cfl_config; ///< CFL configuration

    PatchSchedulerConfig scheduler_conf = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }

    inline Tscal get_constant_G() const {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("gsph::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Riemann Solver Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using RiemannConfig = RiemannConfig<Tvec>;
    RiemannConfig riemann_config;

    inline void set_riemann_iterative(Tscal tol = Tscal{1e-6}, u32 max_iter = 20) {
        riemann_config.set_iterative(tol, max_iter);
    }

    inline void set_riemann_hllc() { riemann_config.set_hllc(); }

    inline void set_riemann_exact(Tscal tol = Tscal{1e-8}, u32 max_iter = 100) {
        riemann_config.set_exact(tol, max_iter);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Riemann Solver Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Reconstruction Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ReconstructConfig = ReconstructConfig<Tvec>;
    ReconstructConfig reconstruct_config;

    inline void set_reconstruct_piecewise_constant() {
        reconstruct_config.set_piecewise_constant();
    }

    inline void set_reconstruct_muscl(
        typename ReconstructConfig::Limiter limiter = ReconstructConfig::Limiter::VanLeer) {
        reconstruct_config.set_muscl(limiter);
    }

    inline bool requires_gradients() const { return reconstruct_config.requires_gradients(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Reconstruction Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Force Formulation Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ForceFormulationConfig = ForceFormulationConfig<Tvec>;
    ForceFormulationConfig force_formulation_config;

    inline void set_force_cha_whitworth() { force_formulation_config.set_cha_whitworth(); }

    inline void set_force_inutsuka_v2() { force_formulation_config.set_inutsuka_v2(); }

    inline bool is_force_inutsuka_v2() const { return force_formulation_config.is_inutsuka_v2(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Force Formulation Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using EOSConfig = shammodels::EOSConfig<Tvec>;
    EOSConfig eos_config;

    inline bool is_eos_adiabatic() const {
        using T = typename EOSConfig::Adiabatic;
        return bool(std::get_if<T>(&eos_config.config));
    }

    inline bool is_eos_isothermal() const {
        using T = typename EOSConfig::Isothermal;
        return bool(std::get_if<T>(&eos_config.config));
    }

    /**
     * @brief Get the adiabatic index (gamma) from the EOS config
     *
     * @return The adiabatic index from Adiabatic or Polytropic EOS, or 1.4 as default
     */
    inline Tscal get_eos_gamma() const {
        using Adiabatic  = typename EOSConfig::Adiabatic;
        using Polytropic = typename EOSConfig::Polytropic;
        if (const auto *eos = std::get_if<Adiabatic>(&eos_config.config)) {
            return eos->gamma;
        } else if (const auto *eos = std::get_if<Polytropic>(&eos_config.config)) {
            return eos->gamma;
        }
        return Tscal{1.4}; // Default for non-gamma EOS types
    }

    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }

    inline void set_eos_isothermal(Tscal cs) { eos_config.set_isothermal(cs); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // EOS Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using BCConfig = shammodels::sph::BCConfig<Tvec>; // Reuse from SPH
    BCConfig boundary_config;

    inline void set_boundary_free() { boundary_config.set_free(); }
    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    /**
     * @brief Set shearing periodic boundary conditions
     *
     * Implements shearing box boundaries (Stone 2010) for simulations
     * of differentially rotating systems (e.g., accretion disks).
     *
     * @param shear_base Base vector for shear periodicity count
     * @param shear_dir Direction of the shear velocity shift
     * @param speed Shear velocity magnitude
     */
    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // External Force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;
    ExtForceConfig ext_force_config{};

    inline void add_ext_force_point_mass(
        Tscal central_mass, Tscal Racc, Tvec central_pos = Tvec{}) {
        ext_force_config.add_point_mass(central_mass, Racc, central_pos);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // External Force Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config
    //////////////////////////////////////////////////////////////////////////////////////////////

    u32 tree_reduction_level  = 3;
    bool use_two_stage_search = true;

    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Tree config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config
    //////////////////////////////////////////////////////////////////////////////////////////////

    bool combined_dtdiv_divcurlv_compute = false; ///< Use the combined dtdivv and divcurlv compute
    /// Factor applied to the smoothing length for neighbors search (and ghost zone size)
    /// @note This value must be larger or equal to htol_up_fine_cycle
    Tscal htol_up_coarse_cycle = 1.1; ///< Factor for neighbors search
    /// Maximum factor of the smoothing length evolution per subcycles
    Tscal htol_up_fine_cycle  = 1.1;
    Tscal epsilon_h           = 1e-6; ///< Convergence criteria for the smoothing length
    u32 h_iter_per_subcycles  = 50;   ///< Maximum number of iterations per subcycle
    u32 h_max_subcycles_count = 100;  ///< Maximum number of subcycles before solver crash

    SmoothingLengthConfig smoothing_length_config;

    inline void set_smoothing_length_density_based() {
        smoothing_length_config.set_density_based();
    }
    inline void set_smoothing_length_density_based_neigh_lim(u32 max_neigh_count) {
        smoothing_length_config.set_density_based_neigh_lim(max_neigh_count);
    }

    bool enable_particle_reordering = false;
    inline void set_enable_particle_reordering(bool enable) { enable_particle_reordering = enable; }
    u64 particle_reordering_step_freq = 1000;
    inline void set_particle_reordering_step_freq(u64 freq) {
        if (freq == 0) {
            shambase::throw_with_loc<std::invalid_argument>(
                "particle_reordering_step_freq cannot be zero");
        }
        particle_reordering_step_freq = freq;
    }

    bool save_dt_to_fields = false;
    inline void set_save_dt_to_fields(bool enable) { save_dt_to_fields = enable; }
    inline bool should_save_dt_to_fields() const { return save_dt_to_fields; }

    bool show_ghost_zone_graph = false;
    inline void set_show_ghost_zone_graph(bool enable) { show_ghost_zone_graph = enable; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver behavior config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    inline bool has_field_uint() const { return is_eos_adiabatic(); }

    inline void print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- GSPH Solver configuration -----");
        logger::raw_ln("gpart_mass  =", gpart_mass);
        riemann_config.print_status();
        reconstruct_config.print_status();
        force_formulation_config.print_status();
        eos_config.print_status();
        logger::raw_ln("--------------------------------------");
    }

    inline void check_config() const {
        // Validate configuration (gpart_mass checked later at runtime)
        // Only check gamma for adiabatic EOS types
        if (is_eos_adiabatic() && get_eos_gamma() <= 1) {
            shambase::throw_with_loc<std::runtime_error>("gamma must be > 1 for adiabatic gas");
        }

        // InutsukaV2 is only wired into update_derivs_iterative()/update_derivs_exact();
        // update_derivs_hllc() would silently fall back to ChaWhitworth otherwise.
        if (force_formulation_config.is_inutsuka_v2() && riemann_config.is_hllc()) {
            shambase::throw_with_loc<std::runtime_error>(
                "InutsukaV2 force formulation is not yet supported with the HLLC Riemann "
                "solver. Use set_riemann_iterative() or set_riemann_exact() instead.");
        }
    }

    inline void check_config_runtime() const {
        // Validate configuration for runtime (called before simulation starts)
        if (gpart_mass <= 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "gpart_mass must be positive. Call set_particle_mass() before evolving.");
        }
        check_config();
    }

    void set_layout(shamrock::patch::PatchDataLayerLayout &pdl);
    void set_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout);
};

namespace shammodels::gsph {

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const CFLConfig<Tscal> &p) {
        j = nlohmann::json{
            {"cfl_cour", p.cfl_cour},
            {"cfl_force", p.cfl_force},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, CFLConfig<Tscal> &p) {
        j.at("cfl_cour").get_to(p.cfl_cour);
        j.at("cfl_force").get_to(p.cfl_force);
    }

    // JSON serialization for SmoothingLengthConfig
    inline void to_json(nlohmann::json &j, const SmoothingLengthConfig &p) {
        if (const SmoothingLengthConfig::DensityBased *conf
            = std::get_if<SmoothingLengthConfig::DensityBased>(&p.config)) {
            j = {
                {"type", "density_based"},
            };

        } else if (
            const SmoothingLengthConfig::DensityBasedNeighLim *conf
            = std::get_if<SmoothingLengthConfig::DensityBasedNeighLim>(&p.config)) {

            j = {
                {"type", "density_based_neigh_lim"},
                {"max_neigh_count", conf->max_neigh_count},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    inline void from_json(const nlohmann::json &j, SmoothingLengthConfig &p) {
        if (j.at("type").get<std::string>() == "density_based") {
            p.config = SmoothingLengthConfig::DensityBased{};
        } else if (j.at("type").get<std::string>() == "density_based_neigh_lim") {
            p.config
                = SmoothingLengthConfig::DensityBasedNeighLim{j.at("max_neigh_count").get<u32>()};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, template<class> class SPHKernel>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        j = nlohmann::json{
            {"solver_type", "gsph"},
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            {"scheduler_config", p.scheduler_conf},
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", p.unit_sys},
            {"riemann_config", p.riemann_config},
            {"reconstruct_config", p.reconstruct_config},
            {"force_formulation_config", p.force_formulation_config},
            {"eos_config", p.eos_config},
            {"boundary_config", p.boundary_config},
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"htol_up_coarse_cycle", p.htol_up_coarse_cycle},
            {"htol_up_fine_cycle", p.htol_up_fine_cycle},
            {"epsilon_h", p.epsilon_h},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},
            {"combined_dtdiv_divcurlv_compute", p.combined_dtdiv_divcurlv_compute},
            {"enable_particle_reordering", p.enable_particle_reordering},
            {"particle_reordering_step_freq", p.particle_reordering_step_freq},
            {"set_save_dt_to_fields", p.save_dt_to_fields},
            {"show_ghost_zone_graph", p.show_ghost_zone_graph},
        };
    }

    template<class Tvec, template<class> class SPHKernel>
    inline void from_json(const nlohmann::json &j, SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = j.at("kernel_id").get<std::string>();
        if (kernel_id != shambase::get_type_name<Tkernel>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid kernel type: expected " + shambase::get_type_name<Tkernel>() + " but got "
                + kernel_id);
        }

        std::string type_id = j.at("type_id").get<std::string>();
        if (type_id != shambase::get_type_name<Tvec>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid vector type: expected " + shambase::get_type_name<Tvec>() + " but got "
                + type_id);
        }

        bool has_used_defaults  = false;
        bool has_updated_config = false;

        auto _get_to_if_contains = [&](const std::string &key, auto &value) {
            shamrock::get_to_if_contains(j, key, value, has_used_defaults);
        };

        _get_to_if_contains("scheduler_config", p.scheduler_conf);
        _get_to_if_contains("gpart_mass", p.gpart_mass);
        _get_to_if_contains("cfl_config", p.cfl_config);
        _get_to_if_contains("unit_sys", p.unit_sys);
        _get_to_if_contains("riemann_config", p.riemann_config);
        _get_to_if_contains("reconstruct_config", p.reconstruct_config);
        _get_to_if_contains("force_formulation_config", p.force_formulation_config);
        _get_to_if_contains("eos_config", p.eos_config);
        _get_to_if_contains("boundary_config", p.boundary_config);
        _get_to_if_contains("tree_reduction_level", p.tree_reduction_level);
        _get_to_if_contains("use_two_stage_search", p.use_two_stage_search);
        _get_to_if_contains("htol_up_coarse_cycle", p.htol_up_coarse_cycle);
        _get_to_if_contains("htol_up_fine_cycle", p.htol_up_fine_cycle);
        _get_to_if_contains("epsilon_h", p.epsilon_h);
        _get_to_if_contains("h_iter_per_subcycles", p.h_iter_per_subcycles);
        _get_to_if_contains("h_max_subcycles_count", p.h_max_subcycles_count);

        if (has_used_defaults || has_updated_config) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "GSPH::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

} // namespace shammodels::gsph
