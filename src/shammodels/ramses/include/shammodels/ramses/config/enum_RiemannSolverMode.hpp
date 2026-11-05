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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Riemann solver mode enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"

namespace shammodels::basegodunov {

    /**
     * @brief Gas Riemann solver used by the Ramses (Godunov) model
     *
     * - Rusanov : local Lax-Friedrichs flux
     * - HLL : Harten-Lax-van Leer flux
     * - HLLC_PVRS : HLLC flux with the wave speeds estimated from the primitive variable
     *   Riemann solver (PVRS) star pressure, following Toro (see shammath::hllc_adiab_toro_flux)
     * - HLLC_Davis : HLLC flux with the Davis (1988) wave speed estimate
     *   (see shammath::hllc_davis_flux)
     */
    enum RiemannSolverMode { Rusanov = 0, HLL = 1, HLLC_PVRS = 2, HLLC_Davis = 3 };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        RiemannSolverMode,
        {{RiemannSolverMode::Rusanov, "rusanov"},
         {RiemannSolverMode::HLL, "hll"},
         {RiemannSolverMode::HLLC_PVRS, "hllc_pvrs"},
         {RiemannSolverMode::HLLC_Davis, "hllc_davis"},
         // legacy name of HLLC_PVRS, only matched when reading (to_json picks the first entry)
         {RiemannSolverMode::HLLC_PVRS, "hllc"}});

} // namespace shammodels::basegodunov
