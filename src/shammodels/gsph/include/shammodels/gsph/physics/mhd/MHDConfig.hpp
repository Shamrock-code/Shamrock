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
 * @file MHDConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Configuration for Magnetohydrodynamics GSPH physics mode
 *
 * Placeholder for future MHD implementation.
 * Will support ideal and resistive MHD.
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>

namespace shammodels::gsph::physics {

    /**
     * @brief Magnetohydrodynamics GSPH configuration
     *
     * Placeholder for future MHD support:
     * - Ideal MHD with constrained transport
     * - Resistive MHD with Ohmic dissipation
     */
    template<class Tvec>
    struct MHDConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal resistivity = Tscal{0.0}; ///< Ohmic resistivity (0 = ideal MHD)

        inline bool is_ideal() const { return resistivity == Tscal{0}; }

        inline void print_status() const {
            logger::raw_ln("  Mode: Magnetohydrodynamics GSPH");
            logger::raw_ln("    resistivity  =", resistivity);
            logger::raw_ln("    ideal MHD    =", is_ideal());
        }
    };

    // JSON serialization
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const MHDConfig<Tvec> &cfg) {
        j = {{"resistivity", cfg.resistivity}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, MHDConfig<Tvec> &cfg) {
        using Tscal     = shambase::VecComponent<Tvec>;
        cfg.resistivity = j.at("resistivity").get<Tscal>();
    }

} // namespace shammodels::gsph::physics
