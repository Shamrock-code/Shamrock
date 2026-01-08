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
 * @file NewtonianConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Configuration for Newtonian GSPH physics mode
 *
 * Standard Newtonian hydrodynamics with Godunov SPH method.
 * Uses kernel-sum density and direct velocity integration.
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::gsph::physics {

    /**
     * @brief Newtonian GSPH configuration
     *
     * Standard Newtonian hydrodynamics:
     * - Kernel-sum density: ρ = Σ_j m_j W(r_ij, h)
     * - Direct velocity integration with leapfrog
     * - Optional grad-h correction for conservation
     */
    template<class Tvec>
    struct NewtonianConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Enable grad-h correction in force computation (Price 2012, Hopkins 2013)
        /// Uses V²/Ω instead of interpolated V² to account for spatial variation of h.
        bool use_grad_h = false;

        inline void print_status() const {
            logger::raw_ln("  Mode: Newtonian GSPH");
            logger::raw_ln("    use_grad_h   =", use_grad_h);
        }
    };

    // JSON serialization
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const NewtonianConfig<Tvec> &cfg) {
        j = {{"use_grad_h", cfg.use_grad_h}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, NewtonianConfig<Tvec> &cfg) {
        if (j.contains("use_grad_h")) {
            cfg.use_grad_h = j.at("use_grad_h").get<bool>();
        }
    }

} // namespace shammodels::gsph::physics
