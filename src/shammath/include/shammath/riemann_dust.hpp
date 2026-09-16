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
 * @file riemann_dust.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Umbrella header pulling in the dust states and all dust Riemann
 *        solvers (HLL, Huang & Bai)
 */

#include "shammath/riemann_common.hpp"
#include "shammath/riemann_dust_hll.hpp"
#include "shammath/riemann_dust_huang_bai.hpp"
