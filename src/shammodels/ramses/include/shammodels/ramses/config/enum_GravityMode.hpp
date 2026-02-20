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
 * @file enum_GravityMode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief Gravity mode enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    enum GravityMode {
        NoGravity = 0,
        CG        = 1, // conjuguate gradient
        PCG       = 2, // preconditioned conjuguate gradient
        BICGSTAB  = 3, // bicgstab
        MULTIGRID = 4  // multigrid
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        GravityMode,
        {{GravityMode::NoGravity, "no_gravity"},
         {GravityMode::CG, "cg"},
         {GravityMode::PCG, "pcg"},
         {GravityMode::BICGSTAB, "bicgstab"},
         {GravityMode::MULTIGRID, "multigrid"}});

} // namespace shammodels::basegodunov
