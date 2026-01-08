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
 * @brief Configuration for Special Relativistic GSPH physics mode
 *
 * Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 * "Special Relativistic Godunov SPH"
 *
 * Key differences from Newtonian:
 * - Conserved variables: S = γHv (momentum), e = γH - P/(Nc²) (energy)
 * - Volume-based density: N = ν/V_p instead of kernel sum
 * - Primitive recovery: Solve quartic for Lorentz factor γ
 * - Uses exact Riemann solver only (Pons et al. 2000)
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::gsph::physics {

    /**
     * @brief Special Relativistic GSPH configuration
     *
     * SR hydrodynamics with exact Riemann solver:
     * - Volume-based density: N = ν/V_p
     * - Conserved variables: S = γHv, e = γH - P/(Nc²)
     * - Newton-Raphson primitive recovery
     */
    template<class Tvec>
    struct SRConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal c_speed      = Tscal{1.0};   ///< Speed of light (default c=1 natural units)
        Tscal eta          = Tscal{1.0};   ///< Smoothing length coefficient
        Tscal c_smooth     = Tscal{2.0};   ///< Kernel expansion for h iteration
        Tscal c_shock      = Tscal{3.0};   ///< Shock detection threshold
        Tscal c_cd         = Tscal{0.2};   ///< Contact discontinuity log(P) threshold
        Tscal tol          = Tscal{1e-10}; ///< Newton-Raphson convergence tolerance
        u32 max_iter       = 100;          ///< Maximum Newton-Raphson iterations
        bool iterative_sml = true;         ///< Iterate smoothing length each step

        inline void print_status() const {
            logger::raw_ln("  Mode: Special Relativistic GSPH");
            logger::raw_ln("    c_speed      =", c_speed);
            logger::raw_ln("    eta          =", eta);
            logger::raw_ln("    c_smooth     =", c_smooth);
            logger::raw_ln("    c_shock      =", c_shock);
            logger::raw_ln("    c_cd         =", c_cd);
            logger::raw_ln("    tol          =", tol);
            logger::raw_ln("    max_iter     =", max_iter);
            logger::raw_ln("    iterative_sml=", iterative_sml);
        }
    };

    // JSON serialization
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const SRConfig<Tvec> &cfg) {
        j = {
            {"c_speed", cfg.c_speed},
            {"eta", cfg.eta},
            {"c_smooth", cfg.c_smooth},
            {"c_shock", cfg.c_shock},
            {"c_cd", cfg.c_cd},
            {"tol", cfg.tol},
            {"max_iter", cfg.max_iter},
            {"iterative_sml", cfg.iterative_sml},
        };
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, SRConfig<Tvec> &cfg) {
        using Tscal       = shambase::VecComponent<Tvec>;
        cfg.c_speed       = j.at("c_speed").get<Tscal>();
        cfg.eta           = j.at("eta").get<Tscal>();
        cfg.c_smooth      = j.at("c_smooth").get<Tscal>();
        cfg.c_shock       = j.at("c_shock").get<Tscal>();
        cfg.c_cd          = j.at("c_cd").get<Tscal>();
        cfg.tol           = j.at("tol").get<Tscal>();
        cfg.max_iter      = j.at("max_iter").get<u32>();
        cfg.iterative_sml = j.at("iterative_sml").get<bool>();
    }

} // namespace shammodels::gsph::physics
