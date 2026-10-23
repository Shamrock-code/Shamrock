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
 * @file enum_AMRInterpoMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief AMR (refinement) prolongation mode enum + json serialization/deserialization
 *
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    enum AMRInterpoMode {
        FIRST_ORDER  = 0, // first order
        SECOND_ORDER = 1, // second order (with Minmod slope limiter + conservative variables)
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        AMRInterpoMode,
        {{AMRInterpoMode::FIRST_ORDER, "amr_first_order"},
         {AMRInterpoMode::SECOND_ORDER, "amr_second_order"}});

} // namespace shammodels::basegodunov
