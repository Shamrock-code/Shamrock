// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverConfig.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamrock/io/units_json.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <stdexcept>

namespace shammodels::basegodunov {

    enum RiemmanSolverMode { Rusanov = 0, HLL = 1, HLLC = 2 };

    enum SlopeMode {
        None        = 0,
        VanLeer_f   = 1,
        VanLeer_std = 2,
        VanLeer_sym = 3,
        Minmod      = 4,
    };

    enum DustRiemannSolverMode {
        NoDust = 0,
        DHLL   = 1, // Dust HLL . This is merely the HLL solver for dust. It's then a Rusanov like
        HB     = 2 // Huang and Bai. Pressureless Riemann solver by Huang and Bai (2022) in Athena++
    };

    enum DragSolverMode {
        NoDrag = 0,
        IRK1   = 1, // Implicit RK1
        IRK2   = 2, // Implicit RK2
        EXPO   = 3  // Matrix exponential
    };

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

    enum GravityMode {
        NoGravity = 0,
        CG        = 1, // conjuguate gradient
        PCG       = 2, // preconditioned conjuguate gradient
        BIGSTAB   = 3, // bigstab
        MULTIGRID = 4  // multigrid
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

    RiemmanSolverMode riemman_config  = HLL;
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
            logger::warn_ln("amr::Config", "the unit system is not set");
            shamunits::Constants<Tscal> ctes{shamunits::UnitSystem<Tscal>{}};
            return ctes.G();
        } else {
            return shamunits::Constants<Tscal>{*unit_sys}.G();
        }
    }
    inline bool is_boundary_periodic() { return true; }
    GravityConfig<Tvec> gravity_config{};
    inline Tscal get_constant_4piG() {
        gravity_config.G_value = get_constant_G();
        return gravity_config.get_fourPiG();
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
            logger::warn_ln("Ramses::SolverConfig", "Dust is experimental");
        }

        if (is_gravity_on()) {
            logger::warn_ln("Ramses::SolverConfig", "Self gravity is experimental");
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
            logger::warn_ln("Ramses::SolverConfig", "Passive scalars are experimental");
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

        nlohmann::json junit;

        j = nlohmann::json{
            {"type_id", shambase::get_type_name<Tvec>()},
            {"RiemmanSolverMode", p.riemman_config},
            {"SlopeMode", p.slope_config},
            {"GravityMode", p.gravity_config.gravity_mode},
            {"PassiveScalarMode", p.npscal_gas_config.npscal_gas},
            {"face_half_time_interpolation", p.face_half_time_interpolation},
            {"eos_gamma", p.eos_gamma},
            {"grid_coord_to_pos_fact", p.grid_coord_to_pos_fact},
            {"DustRiemannSolverMode", p.Csafe},
            {"unit_sys", junit},
            {"time_state", p.time_state}};
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

        std::string type_id = j.at("type_id").get<std::string>();

        if (type_id != shambase::get_type_name<Tvec>()) {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid type to deserialize, wanted " + shambase::get_type_name<Tvec>()
                + " but got " + type_id);
        }

        // actual data stored in the json
        j.at("RiemmanSolverMode").get_to(p.riemman_config);
        j.at("SlopeMode").get_to(p.slope_config);
        j.at("GravityMode").get_to(p.gravity_config.gravity_mode);
        j.at("PassiveScalarMode").get_to(p.npscal_gas_config.npscal_gas);
        j.at("face_half_time_interpolation").get_to(p.face_half_time_interpolation);
        j.at("eos_gamma").get_to(p.eos_gamma);
        j.at("grid_coord_to_pos_fact").get_to(p.grid_coord_to_pos_fact);
        j.at("DustRiemannSolverMode").get_to(p.Csafe);
        from_json_optional(j.at("unit_sys"), p.unit_sys);
        j.at("time_state").get_to(p.time_state);
    }

} // namespace shammodels::basegodunov
