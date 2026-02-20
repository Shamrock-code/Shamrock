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
 * @file enum_DustRiemannSolverMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief Dust Riemann solver mode enum + json serialization/deserialization
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    /// Dust Riemann solver mode enum
    enum DustRiemannSolverMode {
        /// No dust, so no Riemann solver is used
        NoDust = 0,
        /// Dust HLL. This is merely the HLL solver for dust. It's then a Rusanov like
        DHLL = 1,
        /// Huang and Bai. Pressureless Riemann solver by Huang and Bai (2022) in Athena++
        HB = 2
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        DustRiemannSolverMode,
        {{DustRiemannSolverMode::NoDust, "no_dust"},
         {DustRiemannSolverMode::DHLL, "dhll"},
         {DustRiemannSolverMode::HB, "hb"}});

} // namespace shammodels::basegodunov
