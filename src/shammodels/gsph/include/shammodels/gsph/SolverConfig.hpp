// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
 * @brief Shared configuration for the GSPH solver
 *
 * Contains settings shared by ALL physics modes (Newtonian, SR, MHD).
 * Physics-specific settings are in physics/{newtonian,sr,mhd}/ subfolders.
 *
 * What belongs here: EOS, CFL, boundaries, tree params, units
 * What does NOT belong here: Riemann solver (Newtonian), c_smooth (SR), etc.
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
#include "shammodels/sph/config/BCConfig.hpp"
#include "shamrock/io/units_json.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include <nlohmann/json.hpp>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <vector>

namespace shammodels::gsph {

    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;

    template<class Tvec>
    struct SolverStatusVar;

    template<class Tscal>
    struct CFLConfig {
        Tscal cfl_cour  = 0.3;
        Tscal cfl_force = 0.25;
    };

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::SolverStatusVar {
    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time = 0;
    Tscal dt   = 0;
};

template<class Tvec, template<class> class SPHKernel>
struct shammodels::gsph::SolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass{0}; ///< Particle mass (must be set before use)

    CFLConfig<Tscal> cfl_config;

    // ════════════════════════════════════════════════════════════════════════
    // Units Config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

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

    // ════════════════════════════════════════════════════════════════════════
    // Time state (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

    using SolverStatusVar = SolverStatusVar<Tvec>;
    SolverStatusVar time_state;

    inline void set_time(Tscal t) { time_state.time = t; }
    inline void set_next_dt(Tscal dt) { time_state.dt = dt; }
    inline Tscal get_time() const { return time_state.time; }
    inline Tscal get_dt() const { return time_state.dt; }

    // ════════════════════════════════════════════════════════════════════════
    // EOS Config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

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

    inline Tscal get_eos_gamma() const {
        using Adiabatic  = typename EOSConfig::Adiabatic;
        using Polytropic = typename EOSConfig::Polytropic;
        if (const auto *eos = std::get_if<Adiabatic>(&eos_config.config)) {
            return eos->gamma;
        } else if (const auto *eos = std::get_if<Polytropic>(&eos_config.config)) {
            return eos->gamma;
        }
        return Tscal{1.4};
    }

    inline void set_eos_adiabatic(Tscal gamma) { eos_config.set_adiabatic(gamma); }
    inline void set_eos_isothermal(Tscal cs) { eos_config.set_isothermal(cs); }

    // ════════════════════════════════════════════════════════════════════════
    // Boundary Config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

    using BCConfig = shammodels::sph::BCConfig<Tvec>;
    BCConfig boundary_config;

    inline void set_boundary_free() { boundary_config.set_free(); }
    inline void set_boundary_periodic() { boundary_config.set_periodic(); }

    inline void set_boundary_shearing_periodic(i32_3 shear_base, i32_3 shear_dir, Tscal speed) {
        boundary_config.set_shearing_periodic(shear_base, shear_dir, speed);
    }

    // ════════════════════════════════════════════════════════════════════════
    // External Force Config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;
    ExtForceConfig ext_force_config{};

    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Tree config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

    u32 tree_reduction_level  = 3;
    bool use_two_stage_search = true;

    inline void set_tree_reduction_level(u32 level) { tree_reduction_level = level; }
    inline void set_two_stage_search(bool enable) { use_two_stage_search = enable; }

    // ════════════════════════════════════════════════════════════════════════
    // h-iteration config (shared by all physics modes)
    // ════════════════════════════════════════════════════════════════════════

    Tscal htol_up_coarse_cycle = 2.0;
    Tscal htol_up_fine_cycle   = 2.0;
    Tscal epsilon_h            = 1e-6;
    u32 h_iter_per_subcycles   = 50;
    u32 h_max_subcycles_count  = 100;

    /// Smoothing length multiplier for h iteration tolerance (SR uses larger values)
    Tscal c_smooth = Tscal{1.2};

    // ════════════════════════════════════════════════════════════════════════
    // Physics-mode specific (temporary until full decoupling)
    // These are set by the physics mode and used by force kernels
    // ════════════════════════════════════════════════════════════════════════

    Tscal c_speed   = Tscal{1.0};   ///< Speed of light (SR only, c=1 natural units)
    Tscal sr_tol    = Tscal{1e-10}; ///< Newton-Raphson tolerance (SR only)
    u32 sr_max_iter = 100;          ///< Newton-Raphson max iterations (SR only)
    bool use_grad_h = false;        ///< Enable grad-h correction

    /// Ghost layout density field name (SSOT: set by PhysicsMode)
    /// - Newtonian: "density" (mass density ρ)
    /// - SR: "N_labframe" (lab-frame baryon density N)
    std::string density_ghost_field_name = "density";

    // ════════════════════════════════════════════════════════════════════════
    // Utilities
    // ════════════════════════════════════════════════════════════════════════

    inline bool has_field_uint() const { return is_eos_adiabatic(); }

    /// Whether this physics mode uses MUSCL (2nd order) reconstruction
    /// Set to false for piecewise constant (1st order) - default true for Newtonian
    bool use_gradients = true;

    inline bool requires_gradients() const { return use_gradients; }
    inline void set_use_gradients(bool val) { use_gradients = val; }

    /// Per-particle mass field (used by SR mode with volume-based density)
    /// Set by physics mode when it adds the pmass field to the layout
    bool use_pmass_field = false;

    inline bool has_field_pmass() const { return use_pmass_field; }

    inline void set_use_pmass_field(bool val) { use_pmass_field = val; }

    inline void print_status() {
        if (shamcomm::world_rank() != 0) {
            return;
        }
        logger::raw_ln("----- GSPH Solver configuration -----");
        logger::raw_ln("gpart_mass  =", gpart_mass);
        eos_config.print_status();
        logger::raw_ln("--------------------------------------");
    }

    inline void check_config() const {
        if (is_eos_adiabatic() && get_eos_gamma() <= 1) {
            shambase::throw_with_loc<std::runtime_error>("gamma must be > 1 for adiabatic gas");
        }
    }

    inline void check_config_runtime() const {
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

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
        j = nlohmann::json{
            {"time", p.time},
            {"dt", p.dt},
        };
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
        using Tscal = typename SolverStatusVar<Tvec>::Tscal;
        j.at("time").get_to<Tscal>(p.time);
        j.at("dt").get_to<Tscal>(p.dt);
    }

    template<class Tvec, template<class> class SPHKernel>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, SPHKernel> &p) {
        using T       = SolverConfig<Tvec, SPHKernel>;
        using Tkernel = typename T::Kernel;

        std::string kernel_id = shambase::get_type_name<Tkernel>();
        std::string type_id   = shambase::get_type_name<Tvec>();

        nlohmann::json junit;
        to_json_optional(junit, p.unit_sys);

        j = nlohmann::json{
            {"solver_type", "gsph"},
            {"kernel_id", kernel_id},
            {"type_id", type_id},
            {"gpart_mass", p.gpart_mass},
            {"cfl_config", p.cfl_config},
            {"unit_sys", junit},
            {"time_state", p.time_state},
            {"eos_config", p.eos_config},
            {"boundary_config", p.boundary_config},
            {"tree_reduction_level", p.tree_reduction_level},
            {"use_two_stage_search", p.use_two_stage_search},
            {"htol_up_coarse_cycle", p.htol_up_coarse_cycle},
            {"htol_up_fine_cycle", p.htol_up_fine_cycle},
            {"epsilon_h", p.epsilon_h},
            {"h_iter_per_subcycles", p.h_iter_per_subcycles},
            {"h_max_subcycles_count", p.h_max_subcycles_count},
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

        j.at("gpart_mass").get_to(p.gpart_mass);
        j.at("cfl_config").get_to(p.cfl_config);
        from_json_optional(j.at("unit_sys"), p.unit_sys);
        j.at("time_state").get_to(p.time_state);
        j.at("eos_config").get_to(p.eos_config);
        j.at("boundary_config").get_to(p.boundary_config);
        j.at("tree_reduction_level").get_to(p.tree_reduction_level);
        j.at("use_two_stage_search").get_to(p.use_two_stage_search);
        j.at("htol_up_coarse_cycle").get_to(p.htol_up_coarse_cycle);
        j.at("htol_up_fine_cycle").get_to(p.htol_up_fine_cycle);
        j.at("epsilon_h").get_to(p.epsilon_h);
        j.at("h_iter_per_subcycles").get_to(p.h_iter_per_subcycles);
        j.at("h_max_subcycles_count").get_to(p.h_max_subcycles_count);
    }

} // namespace shammodels::gsph
