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
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
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
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include <nlohmann/json.hpp>
#include <shamrock/io/json_std_optional.hpp>
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>
#include <stdexcept>
#include <variant>

namespace shammodels::basegodunov {

    /**
     * @brief alphas is the dust collision rate (the inverse of the stopping time)
     */
    struct DragConfig {
        DragSolverMode drag_solver_config = NoDrag;
        std::vector<f64> alphas;
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
        bool analytical_gravity  = false; // whether to use an external analytical gravity
        Tscal tol                = 1e-6;
        inline Tscal get_tolerance() { return tol; }
        inline bool is_gravity_on() { return gravity_mode != NoGravity; }
    };

    template<class Tvec, class TgridVec>
    struct AMRMode {

        using Tscal = shambase::VecComponent<Tvec>;

        struct None {};

        struct DensityBased {
            Tscal crit_mass;
        };

        struct PseudoGradientBased {
            Tscal error_min;
            Tscal error_max;
        };

        struct JeansLengthBased {
            u32 N_J   = 4;
            Tscal T_0 = 10.;
        };

        struct ShearBased {
            Tscal threshold;
        };

        using mode
            = std::variant<None, DensityBased, PseudoGradientBased, JeansLengthBased, ShearBased>;

        mode config = None{};

        bool old_amr = true;

        void set_refine_none() { config = None{}; }
        void set_refine_density_based(Tscal crit_mass) { config = DensityBased{crit_mass}; }
        void set_refine_pseudo_gradient_based(Tscal error_min, Tscal error_max) {
            config = PseudoGradientBased{error_min, error_max};
        }

        void set_refine_jeans_length_based(u32 N_J, Tscal T_0) {
            config = JeansLengthBased{N_J, T_0};
        }

        void set_refine_shear_based(Tscal thresh) { config = ShearBased{thresh}; }

        bool need_level_zero_compute() { return !old_amr; }
        bool need_amr_level_compute() { return !old_amr; }
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

template<class Tvec, class TgridVec>
struct shammodels::basegodunov::SolverConfig {

    using Tscal = shambase::VecComponent<Tvec>;

    Tscal eos_gamma = 5. / 3.;

    Tscal grid_coord_to_pos_fact = 1;

    static constexpr u32 NsideBlockPow = 1;
    using AMRBlock                     = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

    inline void set_eos_gamma(Tscal gamma) { eos_gamma = gamma; }

    RiemannSolverMode riemann_config  = HLL;
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
    inline bool is_coordinate_field_required() { return gravity_config.analytical_gravity; }

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

    PatchSchedulerConfig scheduler_conf = {};

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (config)
    //////////////////////////////////////////////////////////////////////////////////////////////

    Tscal Csafe = 0.9;

    //////////////////////////////////////////////////////////////////////////////////////////////
    // CFL Configuration (END)
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

            shamrock::experimental_feature_check(
                shambase::format(
                    "self gravity mode is not enabled but gravity mode is set to {} (> 0 whith 0 "
                    "== "
                    "NoGravity mode)",
                    mode));
        }

        if (!(eos_gamma > 1.0)) {
            shambase::throw_with_loc<std::invalid_argument>(
                shambase::format("Gamma must be > 1, currently Gamma = {}", eos_gamma));
        }

        if (is_gas_passive_scalar_on()) {
            ON_RANK_0(logger::warn_ln("Ramses::SolverConfig", "Passive scalars are experimental"));
            shamrock::experimental_feature_check(
                shambase::format(
                    "gas passive scalars mode is not enabled but gas passive scalars mode is set "
                    "to {}"
                    "> 0",
                    npscal_gas_config.npscal_gas));
        }

        if (!amr_mode.old_amr) {
            shamrock::experimental_feature_check("new AMR is experimental");
        }
    }

    void set_layout(shamrock::patch::PatchDataLayerLayout &pdl);
};

namespace shammodels::basegodunov {

    inline void to_json(nlohmann::json &j, const BCConfig::GhostType &e) {
        switch (e) {
        case BCConfig::GhostType::Periodic  : j = "periodic"; break;
        case BCConfig::GhostType::Reflective: j = "reflective"; break;
        case BCConfig::GhostType::Outflow   : j = "outflow"; break;
        default:
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + std::to_string(static_cast<int>(e)));
        }
    }

    inline void from_json(const nlohmann::json &j, BCConfig::GhostType &e) {
        const std::string type = j.get<std::string>();
        if (type == "periodic") {
            e = BCConfig::GhostType::Periodic;
        } else if (type == "reflective") {
            e = BCConfig::GhostType::Reflective;
        } else if (type == "outflow") {
            e = BCConfig::GhostType::Outflow;
        } else {
            shambase::throw_with_loc<std::runtime_error>(
                "Invalid BCConfig::GhostType value: " + type);
        }
    }

    inline void to_json(nlohmann::json &j, const BCConfig &p) {
        j = nlohmann::json{
            {"ghost_type_x", p.ghost_type_x},
            {"ghost_type_y", p.ghost_type_y},
            {"ghost_type_z", p.ghost_type_z}};
    }

    inline void from_json(const nlohmann::json &j, BCConfig &p) {
        j.at("ghost_type_x").get_to(p.ghost_type_x);
        j.at("ghost_type_y").get_to(p.ghost_type_y);
        j.at("ghost_type_z").get_to(p.ghost_type_z);
    }

    inline void to_json(nlohmann::json &j, const DragConfig &p) {
        j = nlohmann::json{
            {"drag_solver", p.drag_solver_config},
            {"alphas", p.alphas},
            {"enable_frictional_heating", p.enable_frictional_heating}};
    }

    inline void from_json(const nlohmann::json &j, DragConfig &p) {
        j.at("drag_solver").get_to(p.drag_solver_config);
        j.at("alphas").get_to(p.alphas);
        j.at("enable_frictional_heating").get_to(p.enable_frictional_heating);
    }

    template<class Tvec, class TgridVec>
    inline void amr_config_to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
        using AMR = AMRMode<Tvec, TgridVec>;

        if (std::holds_alternative<typename AMR::None>(p.config)) {
            j = {{"type", "none"}};
        } else if (const auto *cfg = std::get_if<typename AMR::DensityBased>(&p.config)) {
            j = {{"type", "density_based"}, {"crit_mass", cfg->crit_mass}};
        } else if (const auto *cfg = std::get_if<typename AMR::PseudoGradientBased>(&p.config)) {
            j
                = {{"type", "pseudo_gradient_based"},
                   {"error_min", cfg->error_min},
                   {"error_max", cfg->error_max}};
        } else if (const auto *cfg = std::get_if<typename AMR::JeansLengthBased>(&p.config)) {
            j = {{"type", "jeans_length_based"}, {"N_J", cfg->N_J}, {"T_0", cfg->T_0}};
        } else if (const auto *cfg = std::get_if<typename AMR::ShearBased>(&p.config)) {
            j = {{"type", "shear_based"}, {"threshold", cfg->threshold}};
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec, class TgridVec>
    inline void amr_config_from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        const std::string type = j.at("type").get<std::string>();
        if (type == "none") {
            p.set_refine_none();
        } else if (type == "density_based") {
            p.set_refine_density_based(j.at("crit_mass").get<Tscal>());
        } else if (type == "pseudo_gradient_based") {
            p.set_refine_pseudo_gradient_based(
                j.at("error_min").get<Tscal>(), j.at("error_max").get<Tscal>());
        } else if (type == "jeans_length_based") {
            p.set_refine_jeans_length_based(j.at("N_J").get<u32>(), j.at("T_0").get<Tscal>());
        } else if (type == "shear_based") {
            p.set_refine_shear_based(j.at("threshold").get<Tscal>());
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid AMR mode type: " + type);
        }
    }

    template<class Tvec, class TgridVec>
    inline void to_json(nlohmann::json &j, const AMRMode<Tvec, TgridVec> &p) {
        nlohmann::json config_j;
        amr_config_to_json(config_j, p);
        j = nlohmann::json{{"old_amr", p.old_amr}, {"config", config_j}};
    }

    template<class Tvec, class TgridVec>
    inline void from_json(const nlohmann::json &j, AMRMode<Tvec, TgridVec> &p) {
        j.at("old_amr").get_to(p.old_amr);
        amr_config_from_json(j.at("config"), p);
    }

    /**
     * @brief Serialize a SolverConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The SolverConfig to serialize
     */
    template<class Tvec, class TgridVec>
    void to_json(nlohmann::json &j, const SolverConfig<Tvec, TgridVec> &p);
    /**
     * @brief Deserializes a SolverConfig object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The SolverConfig object to populate.
     */
    template<class Tvec, class TgridVec>
    void from_json(const nlohmann::json &j, SolverConfig<Tvec, TgridVec> &p);

} // namespace shammodels::basegodunov
