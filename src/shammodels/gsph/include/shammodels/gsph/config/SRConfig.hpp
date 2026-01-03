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
 * @file SRConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Configuration for Special Relativistic GSPH (SR-GSPH)
 *
 * Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 * "Special Relativistic Godunov SPH"
 *
 * Key differences from standard GSPH:
 * - Canonical conserved variables: S = γHv (momentum), e = γH - P/(Nc²) (energy)
 * - Volume-based density: N = ν/V_p instead of kernel sum
 * - Primitive recovery: Solve quartic for Lorentz factor γ
 * - Tangential velocity support for 1D tests with transverse motion
 *
 * Uses exact Riemann solver with Newton-Raphson iteration following:
 * - Pons et al. (2000) "Exact solution of Riemann problem"
 * - Kitajima et al. (2025) Godunov SPH with exact Riemann solver
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::gsph {

    template<class Tvec>
    struct SRConfig;

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::SRConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief No special relativity (standard Newtonian GSPH)
     */
    struct None {};

    /**
     * @brief Special Relativistic GSPH with exact Riemann solver
     *
     * Uses Newton-Raphson iteration to find exact P* following:
     * - Pons et al. (2000) "Exact solution of Riemann problem with
     *   non-zero tangential velocities in relativistic hydrodynamics"
     * - Kitajima et al. (2025) Godunov SPH with exact Riemann solver
     */
    struct SR {
        Tscal c_speed      = Tscal{1.0};   ///< Speed of light (default c=1 natural units)
        Tscal eta          = Tscal{1.0};   ///< Smoothing length coefficient
        Tscal c_smooth     = Tscal{2.0};   ///< Kernel expansion for h iteration
        Tscal c_shock      = Tscal{3.0};   ///< Shock detection threshold
        Tscal c_cd         = Tscal{0.2};   ///< Contact discontinuity log(P) threshold
        Tscal tol          = Tscal{1e-10}; ///< Newton-Raphson convergence tolerance
        u32 max_iter       = 100;          ///< Maximum Newton-Raphson iterations
        bool iterative_sml = true;         ///< Iterate smoothing length each step
    };

    using Variant = std::variant<None, SR>;

    Variant config = None{};

    void set(Variant v) { config = v; }

    /**
     * @brief Check if special relativity is enabled
     */
    inline bool is_sr_enabled() const { return bool(std::get_if<SR>(&config)); }

    /**
     * @brief Get speed of light (returns 0 if SR not enabled)
     */
    inline Tscal get_c_speed() const {
        if (const auto *sr = std::get_if<SR>(&config)) {
            return sr->c_speed;
        }
        return Tscal{0};
    }

    /**
     * @brief Get smoothing length coefficient eta (throws if SR not enabled)
     */
    inline Tscal get_eta() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("eta not available: SR not enabled");
        }
        return sr->eta;
    }

    /**
     * @brief Get kernel expansion for h iteration c_smooth (throws if SR not enabled)
     */
    inline Tscal get_c_smooth() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("c_smooth not available: SR not enabled");
        }
        return sr->c_smooth;
    }

    /**
     * @brief Get shock detection threshold c_shock (throws if SR not enabled)
     */
    inline Tscal get_c_shock() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("c_shock not available: SR not enabled");
        }
        return sr->c_shock;
    }

    /**
     * @brief Get contact discontinuity threshold c_cd (throws if SR not enabled)
     */
    inline Tscal get_c_cd() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("c_cd not available: SR not enabled");
        }
        return sr->c_cd;
    }

    /**
     * @brief Get Newton-Raphson tolerance (throws if SR not enabled)
     */
    inline Tscal get_tol() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("tol not available: SR not enabled");
        }
        return sr->tol;
    }

    /**
     * @brief Get max Newton-Raphson iterations (throws if SR not enabled)
     */
    inline u32 get_max_iter() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("max_iter not available: SR not enabled");
        }
        return sr->max_iter;
    }

    /**
     * @brief Get SR config parameters (throws if SR not enabled)
     */
    inline const SR &get_sr_config() const {
        const auto *sr = std::get_if<SR>(&config);
        if (!sr) {
            shambase::throw_with_loc<std::runtime_error>("SR config not available: SR not enabled");
        }
        return *sr;
    }

    /**
     * @brief Check if SR requires conserved momentum field S
     */
    inline bool has_S_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if SR requires canonical energy field e
     */
    inline bool has_e_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if SR requires tangent momentum field S_t
     */
    inline bool has_S_t_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if SR requires Lorentz factor field
     */
    inline bool has_gamma_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if SR requires enthalpy field H
     */
    inline bool has_enthalpy_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if SR requires lab-frame density N
     */
    inline bool has_N_field() const { return is_sr_enabled(); }

    /**
     * @brief Check if iterative smoothing length is enabled for SR mode
     * Returns false if SR is not enabled or if iterative_sml is false
     */
    inline bool is_iterative_sml_enabled() const {
        if (const auto *sr = std::get_if<SR>(&config)) {
            return sr->iterative_sml;
        }
        return false;
    }

    inline void print_status() {
        logger::raw_ln("--- SR config");

        if (std::get_if<None>(&config)) {
            logger::raw_ln("  SR Mode: None (Standard Newtonian GSPH)");
        } else if (const auto *sr = std::get_if<SR>(&config)) {
            logger::raw_ln("  SR Mode: Special Relativistic GSPH (Exact solver)");
            logger::raw_ln("  c_speed      =", sr->c_speed);
            logger::raw_ln("  eta          =", sr->eta);
            logger::raw_ln("  c_smooth     =", sr->c_smooth);
            logger::raw_ln("  c_shock      =", sr->c_shock);
            logger::raw_ln("  c_cd         =", sr->c_cd);
            logger::raw_ln("  tol          =", sr->tol);
            logger::raw_ln("  max_iter     =", sr->max_iter);
            logger::raw_ln("  iterative_sml=", sr->iterative_sml);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("-------------");
    }
};

namespace shammodels::gsph {

    /**
     * @brief Serialize SRConfig to JSON
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SRConfig<Tvec> &p) {
        using T    = SRConfig<Tvec>;
        using None = typename T::None;
        using SR   = typename T::SR;

        if (std::get_if<None>(&p.config)) {
            j = {{"sr_type", "none"}};
        } else if (const auto *sr = std::get_if<SR>(&p.config)) {
            j = {
                {"sr_type", "sr"},
                {"c_speed", sr->c_speed},
                {"eta", sr->eta},
                {"c_smooth", sr->c_smooth},
                {"c_shock", sr->c_shock},
                {"c_cd", sr->c_cd},
                {"tol", sr->tol},
                {"max_iter", sr->max_iter},
                {"iterative_sml", sr->iterative_sml},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    /**
     * @brief Deserialize SRConfig from JSON
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SRConfig<Tvec> &p) {
        using T     = SRConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("sr_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field sr_type found in json");
        }

        std::string sr_type;
        j.at("sr_type").get_to(sr_type);

        using None = typename T::None;
        using SR   = typename T::SR;

        if (sr_type == "none") {
            p.set(None{});
        } else if (sr_type == "sr") {
            p.set(
                SR{
                    j.at("c_speed").get<Tscal>(),
                    j.at("eta").get<Tscal>(),
                    j.at("c_smooth").get<Tscal>(),
                    j.at("c_shock").get<Tscal>(),
                    j.at("c_cd").get<Tscal>(),
                    j.at("tol").get<Tscal>(),
                    j.at("max_iter").get<u32>(),
                    j.at("iterative_sml").get<bool>(),
                });
        } else {
            shambase::throw_unimplemented("Unknown SR type: " + sr_type);
        }
    }

} // namespace shammodels::gsph
