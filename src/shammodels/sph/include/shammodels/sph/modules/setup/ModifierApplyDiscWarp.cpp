

// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ModifierApplyDiscWarp.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "ModifierApplyDiscWarp.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec>
shamrock::patch::PatchData shammodels::sph::modules::ModifierApplyDiscWarp<Tvec>::next_n(u32 nmax) {

    shamrock::patch::PatchData tmp = parent->next_n(nmax);
    // apply warp on the content of patchdata tmp

    return tmp;
}
