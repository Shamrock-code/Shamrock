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
 * @file enum_AMRInterpMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AMR (refinement) prolongation mode enum + json serialization/deserialization
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    enum AMRInterpMode {
        FIRST_ORDER  = 0, // first order
        SECOND_ORDER = 1, // second order (with Minmod slope limiter + conservative variables)
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        AMRInterpMode,
        {{AMRInterpMode::FIRST_ORDER, "amr_first_order"},
         {AMRInterpMode::SECOND_ORDER, "amr_second_order"}});

} // namespace shammodels::basegodunov
