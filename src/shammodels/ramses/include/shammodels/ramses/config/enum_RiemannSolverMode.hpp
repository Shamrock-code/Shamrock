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
 * @file enum_RiemannSolverMode.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Riemann solver mode enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    enum RiemannSolverMode { Rusanov = 0, HLL = 1, HLLC = 2 };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        RiemannSolverMode,
        {{RiemannSolverMode::Rusanov, "rusanov"},
         {RiemannSolverMode::HLL, "hll"},
         {RiemannSolverMode::HLLC, "hllc"}});

} // namespace shammodels::basegodunov
