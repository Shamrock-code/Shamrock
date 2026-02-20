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
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Benoit Commercon (benoit.commercon@ens-lyon.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/config/enum_DragSolverMode.hpp"
#include "shammodels/ramses/config/enum_DustRiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_GravityMode.hpp"
#include "shammodels/ramses/config/enum_RiemannSolverMode.hpp"
#include "shammodels/ramses/config/enum_SlopeMode.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/io/json_print_diff.hpp"
#include "shamrock/io/json_std_optional.hpp"
#include "shamrock/io/json_utils.hpp"
#include "shamrock/io/units_json.hpp"
#include <shamrock/io/json_std_optional.hpp>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <stdexcept>

namespace shammodels::basegodunov {

    /**
     * @brief alphas is the dust collision rate (the inverse of the stopping time)
     */
    struct DragConfig {
        DragSolverMode drag_solver_config = NoDrag;
        std::vector<f32> alphas;
        bool enable_frictional_heating
            = false; // 0 to turn off and 1 when all dissipation is deposited to the gas
    };

    struct DustConfig {
        DustRiemannSolverMode dust_riemann_config = NoDust;
        u32 ndust                                 = 0;

        inline bool is_dust_on() {
            if (dust_riemann_config != NoDust) {

                if (ndust == 0) {
                    throw shambase::make_except_with_loc<std::runtime_error>(
                        "Dust is on with ndust == 0");
                }
                return true;
            }
            return false;
        }
    };
    /**
     * @brief Npscal_gas is the number of gas passive scalars
     */
    struct PassiveScalarGasConfig {
        u32 npscal_gas = 0;

        inline bool is_gas_passive_scalar_on() { return npscal_gas > 0; }
    };

    template<class Tvec>
    struct GravityConfig {
        using Tscal              = shambase::VecComponent<Tvec>;
        GravityMode gravity_mode = NoGravity;
        Tscal tol                = 1e-6;
        inline Tscal get_tolerance() { return tol; }
        inline bool is_gravity_on() {
            if (gravity_mode != NoGravity) {
                return true;
            }
            return false;
        }
    };

    template<class Tvec>
    struct SolverStatusVar;

    template<class Tvec, class TgridVec>
    struct AMRMode {

        using Tscal = shambase::VecComponent<Tvec>;

        struct None {};
        struct DensityBased {
            Tscal crit_mass;
        };

        using mode = std::variant<None, DensityBased>;

        mode config = None{};

        void set_refine_none() { config = None{}; }
        void set_refine_density_based(Tscal crit_mass) { config = DensityBased{crit_mass}; }
    };

    struct BCConfig {
        enum class GhostType { Periodic = 0, Reflective = 1, Outflow = 2 };

        GhostType ghost_type_x = GhostType::Periodic;
        GhostType ghost_type_y = GhostType::Periodic;
        GhostType ghost_type_z = GhostType::Periodic;

        GhostType get_x() const { return ghost_type_x; }
        GhostType get_y() const { return ghost_type_y; }
        GhostType get_z() const { return ghost_type_z; }

        void set_x(GhostType ghost_type) { ghost_type_x = ghost_type; }
        void set_y(GhostType ghost_type) { ghost_type_y = ghost_type; }
        void set_z(GhostType ghost_type) { ghost_type_z = ghost_type; }
    };

    template<class Tvec, class TgridVec>
    struct SolverConfig;

}; // namespace shammodels::basegodunov

template<class Tvec>
struct shammodels::basegodunov::SolverStatusVar {

    /// The type of the scalar used to represent the quantities
    using Tscal = shambase::VecComponent<Tvec>;

    Tscal time = 0; ///< Current time
    Tscal dt   = 0; ///< Current time step
};

template<class Tvec, class TgridVec>
struct shammodels::basegodunov::SolverConfig {

    using Tscal = shambase::VecComponent<Tvec>;

    Tscal eos_gamma = 5. / 3.;

    Tscal grid_coord_to_pos_fact = 1;

    static constexpr u32 NsideBlockPow = 1;
    using AMRBlock                     = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

    inline void set_eos_gamma(Tscal gamma) { eos_gamma = gamma; }

    RiemannSolverMode Riemann_config  = HLL;
    SlopeMode slope_config            = VanLeer_sym;
    bool face_half_time_interpolation = true;

    inline bool should_compute_rho_mean() { return is_gravity_on() && is_boundary_periodic(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config
    //////////////////////////////////////////////////////////////////////////////////////////////

    DustConfig dust_config{};
    DragConfig drag_config{};

    inline bool is_dust_on() { return dust_config.is_dust_on(); }
    // get alpha values from user
    // alphas is the dust collision rate (the inverse of the stopping time)
    inline void set_alphas_static(f32 alpha_values) {
        StackEntry stack_lock{};
        drag_config.alphas.push_back(alpha_values);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Dust config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    BCConfig bc_config{};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gas passive scalars config
    //////////////////////////////////////////////////////////////////////////////////////////////

    PassiveScalarGasConfig npscal_gas_config{};

    inline bool is_gas_passive_scalar_on() { return npscal_gas_config.is_gas_passive_scalar_on(); }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gas passive scalars config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gravity config
    //////////////////////////////////////////////////////////////////////////////////////////////
    inline Tscal get_constant_G() {
        if (!unit_sys) {
            ON_RANK_0(logger::warn_ln("amr::Config", "the unit system is not set"));
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }
    inline bool is_boundary_periodic() { return true; }
    GravityConfig<Tvec> gravity_config{};
    inline Tscal get_constant_4piG() {
        auto scal_G = get_constant_G();
        return 4 * M_PI * scal_G;
    }
    inline Tscal get_grav_tol() { return gravity_config.get_tolerance(); }
    inline bool is_gravity_on() { return gravity_config.is_gravity_on(); }

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Gravity config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// AMR refinement mode
    AMRMode<Tvec, TgridVec> amr_mode = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// The unit system of the simulation
    std::optional<shamunits::UnitSystem<Tscal>> unit_sys = {};

    /// Set the unit system of the simulation
    inline void set_units(shamunits::UnitSystem<Tscal> new_sys) { unit_sys = new_sys; }
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Units Config (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables
    //////////////////////////////////////////////////////////////////////////////////////////////

    /// Alias to SolverStatusVar type
    using SolverStatusVar = SolverStatusVar<Tvec>;
    /// The time sate of the simulation
    SolverStatusVar time_state;
    /// Set the current time
    inline void set_time(Tscal t) { time_state.time = t; }
    /// Set the time step for the next iteration
    inline void set_next_dt(Tscal dt) { time_state.dt = dt; }
    /// Get the current time
    inline Tscal get_time() { return time_state.time; }
    /// Get the time step for the next iteration
    inline Tscal get_dt() { return time_state.dt; }

    Tscal Csafe = 0.9;
    //////////////////////////////////////////////////////////////////////////////////////////////
    // Solver status variables (END)
    //////////////////////////////////////////////////////////////////////////////////////////////

    inline void check_config() {
        if (grid_coord_to_pos_fact <= 0) {
            shambase::throw_with_loc<std::runtime_error>(shambase::format(
                "grid_coord_to_pos_fact must be > 0, got {}", grid_coord_to_pos_fact));
        }

        if (is_dust_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Dust is experimental"));
        }

        if (is_gravity_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Self gravity is experimental"));
            u32 mode = gravity_config.gravity_mode;

            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "self gravity mode is not enabled but gravity mode is set to {} (> 0 whith 0 "
                    "== "
                    "NoGravity mode)",
                    mode));
            }
        }

        if (!(eos_gamma > 1.0)) {
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("Gamma must be > 1, currently Gamma = {}", eos_gamma));
        }

        if (is_gas_passive_scalar_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Passive scalars are experimental"));
            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "gas passive scalars mode is not enabled but gas passive scalars mode is set "
                    "to {}"
                    "> 0",
                    npscal_gas_config.npscal_gas));
            }
        }
    }
};

namespace shammodels::basegodunov {

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SolverStatusVar<Tvec> &p) {
        j = nlohmann::json{{"time", p.time}, {"dt", p.dt}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SolverStatusVar<Tvec> &p) {
        using Tscal = typename SolverStatusVar<Tvec>::Tscal;
        j.at("time").get_to<Tscal>(p.time);
        j.at("dt").get_to<Tscal>(p.dt);
    }

    /**
     * @brief Serialize a SolverConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The SolverConfig to serialize
     */
    template<class Tvec, class TgridVec>
    inline void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p) {

        j = nlohmann::json{
            {"type_id", shambase::get_type_name<Tvec>()},
            {"courant_safety_factor", p.Csafe},
            {"dust_riemann_solver", p.dust_config.dust_riemann_config},
            {"eos_gamma", p.eos_gamma},
            {"face_half_time_interpolation", p.face_half_time_interpolation},
            {"gravity_solver", p.gravity_config.gravity_mode},
            {"grid_coord_to_pos_fact", p.grid_coord_to_pos_fact},
            {"hydro_riemann_solver", p.Riemann_config},
            {"passive_scalar_mode", p.npscal_gas_config.npscal_gas},
            {"slope_limiter", p.slope_config},
            {"time_state", p.time_state},
            {"unit_sys", p.unit_sys}};
    }

    /**
     * @brief Deserializes a SolverConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverConfig object to populate.
     */
    template<class Tvec, class TgridVec>
    inline void from_json(const nlohmann::json &j, SolverConfig<Tvec, TgridVec> &p) {
        using T = SolverConfig<Tvec, TgridVec>;

        if (j.contains("type_id")) {

            std::string type_id = j.at("type_id").get<std::string>();

            if (type_id != shambase::get_type_name<Tvec>()) {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                    + " but got " + type_id);
            }
        }

        bool has_used_defaults  = false;
        bool has_updated_config = false;

        auto get_to_if_contains = [&](const std::string &key, auto &value) {
            if (j.contains(key)) {
                j.at(key).get_to(value);
            } else {
                has_used_defaults = true;
            }
        };

        // actual data stored in the json
        get_to_if_contains("courant_safety_factor", p.Csafe);
        get_to_if_contains("dust_riemann_solver", p.dust_config.dust_riemann_config);
        get_to_if_contains("eos_gamma", p.eos_gamma);
        get_to_if_contains("face_half_time_interpolation", p.face_half_time_interpolation);
        get_to_if_contains("gravity_solver", p.gravity_config.gravity_mode);
        get_to_if_contains("grid_coord_to_pos_fact", p.grid_coord_to_pos_fact);
        get_to_if_contains("hydro_riemann_solver", p.Riemann_config);
        get_to_if_contains("passive_scalar_mode", p.npscal_gas_config.npscal_gas);
        get_to_if_contains("slope_limiter", p.slope_config);
        get_to_if_contains("time_state", p.time_state);
        get_to_if_contains("unit_sys", p.unit_sys);

        if (has_used_defaults) {
            if (shamcomm::world_rank() == 0) {
                logger::info_ln(
                    "Ramses::SolverConfig",
                    shamrock::log_json_changes(p, j, has_used_defaults, has_updated_config));
            }
        }
    }

} // namespace shammodels::basegodunov
