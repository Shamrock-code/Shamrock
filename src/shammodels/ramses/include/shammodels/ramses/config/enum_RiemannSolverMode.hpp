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
     * - HLLC : HLLC flux with the pressure based (PVRS) wave speed estimate from Toro
     *   (see shammath::hllc_adiab_toro_flux)
     * - HLLCDavis : HLLC flux with the Davis (1988) wave speed estimate
     *   (see shammath::hllc_davis_flux)
     */
    enum RiemannSolverMode { Rusanov = 0, HLL = 1, HLLC = 2, HLLCDavis = 3 };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        RiemannSolverMode,
        {{RiemannSolverMode::Rusanov, "rusanov"},
         {RiemannSolverMode::HLL, "hll"},
         {RiemannSolverMode::HLLC, "hllc"},
         {RiemannSolverMode::HLLCDavis, "hllc_davis"}});

} // namespace shammodels::basegodunov
