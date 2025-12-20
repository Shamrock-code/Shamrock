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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
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
#include "shammodels/gsph/config/ReconstructConfig.hpp"
#include "shammodels/gsph/config/RiemannConfig.hpp"
#include "shammodels/sph/config/BCConfig.hpp" // Reuse boundary conditions from SPH
#include "shamrock/io/units_json.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
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
     * @brief Solver status variables for GSPH
     *
     * @tparam Tvec the type of the vector used to represent the particles
     */
    template<class Tvec>
    struct SolverStatusVar;

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

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::SolverStatusVar {
    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time = 0; ///< Current time
    Tscal dt   = 0; ///< Current time step
};

template<class Tvec, template<class> class SPHKernel>
struct shammodels::gsph::SolverConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    using Kernel             = SPHKernel<Tscal>;
    using u_morton           = u32;

    using RTree = shamtree::CompressedLeafBVH<u_morton, Tvec, 3>;

    static constexpr Tscal Rkern = Kernel::Rkern;

    Tscal gpart_mass{0};      ///< The mass of each gas particle (must be set before use)
    Tscal gamma = Tscal{1.4}; ///< Adiabatic index (for ideal gas EOS)

    CFLConfig<Tscal> cfl_config; ///< CFL configuration

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
    // Solver status variables
    //////////////////////////////////////////////////////////////////////////////////////////////

    using SolverStatusVar = SolverStatusVar<Tvec>;
    SolverStatusVar time_state;

    inline void set_time(Tscal t) { time_state.time = t; }
    inline void set_next_dt(Tscal dt) { time_state.dt = dt; }
    inline Tscal get_time() const { return time_state.time; }
    inline Tscal get_dt() const { return time_state.dt; }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Riemann Solver Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using RiemannConfig = RiemannConfig<Tvec>;
    RiemannConfig riemann_config;

    inline void set_riemann_iterative(Tscal tol = Tscal{1e-6}, u32 max_iter = 20) {
        riemann_config.set_iterative(tol, max_iter);
    }

    inline void set_riemann_exact(Tscal tol = Tscal{1e-8}) { riemann_config.set_exact(tol); }

    inline void set_riemann_hllc() { riemann_config.set_hllc(); }

    inline void set_riemann_roe(Tscal entropy_fix = Tscal{0.1}) {
        riemann_config.set_roe(entropy_fix);
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

    inline void set_eos_adiabatic(Tscal _gamma) {
        gamma = _gamma;
        eos_config.set_adiabatic(_gamma);
    }

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

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Boundary Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // External Force Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    using ExtForceConfig = shammodels::ExtForceConfig<Tvec>;
    ExtForceConfig ext_force_config{};

    inline void add_ext_force_point_mass(Tscal central_mass, Tscal Racc) {
        ext_force_config.add_point_mass(central_mass, Racc);
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

    Tscal htol_up_coarse_cycle = 1.1;  ///< Factor for neighbors search
    Tscal htol_up_fine_cycle   = 1.1;  ///< Max smoothing length evolution per subcycle
    Tscal epsilon_h            = 1e-6; ///< Convergence criteria for smoothing length
    u32 h_iter_per_subcycles   = 50;   ///< Max iterations per subcycle
    u32 h_max_subcycles_count  = 100;  ///< Max subcycles before crash

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
        logger::raw_ln("gamma       =", gamma);
        riemann_config.print_status();
        reconstruct_config.print_status();
        eos_config.print_status();
        logger::raw_ln("--------------------------------------");
    }

    inline void check_config() const {
        // Validate configuration (gpart_mass checked later at runtime)
        if (gamma <= 1) {
            shambase::throw_with_loc<std::runtime_error>("gamma must be > 1 for ideal gas");
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
            {"gamma", p.gamma},
            {"cfl_config", p.cfl_config},
            {"unit_sys", junit},
            {"time_state", p.time_state},
            {"riemann_config", p.riemann_config},
            {"reconstruct_config", p.reconstruct_config},
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
        j.at("gamma").get_to(p.gamma);
        j.at("cfl_config").get_to(p.cfl_config);
        from_json_optional(j.at("unit_sys"), p.unit_sys);
        j.at("time_state").get_to(p.time_state);
        j.at("riemann_config").get_to(p.riemann_config);
        j.at("reconstruct_config").get_to(p.reconstruct_config);
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
