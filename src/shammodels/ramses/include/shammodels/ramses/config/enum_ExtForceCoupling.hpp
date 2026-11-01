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
 * @file enum_ExtForceCoupling.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief External forces time coupling enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    /**
     * @brief Time quadrature used to couple the external forces source term to the solver
     *
     * Both are second order in time.
     */
    enum ExtForceCoupling {
        /// Explicit midpoint rule: the forces are evaluated once, at t^{n+1/2}, from the time
        /// centered state (density, momentum, ...), and applied over the whole time step
        Midpoint = 0,
        /// Trapezoidal rule (time centred source term, as in RAMSES): the forces are evaluated at
        /// t^n and t^{n+1}, and the average of the two source terms is applied over the whole
        /// time step
        Trapezoidal = 1
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        ExtForceCoupling,
        {{ExtForceCoupling::Midpoint, "midpoint"}, {ExtForceCoupling::Trapezoidal, "trapezoidal"}});

} // namespace shammodels::basegodunov
