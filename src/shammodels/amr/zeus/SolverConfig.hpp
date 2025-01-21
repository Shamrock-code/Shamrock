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
 * @file SolverConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/amr/zeus/modules/SolverStorage.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::zeus {

    template<class Tvec, class TgridVec>
    struct SolverConfig {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal eos_gamma = 5. / 3.;

        Tscal grid_coord_to_pos_fact = 1;

        static constexpr u32 NsideBlockPow = 1;
        using AMRBlock                     = amr::AMRBlock<Tvec, TgridVec, NsideBlockPow>;

        inline void set_eos_gamma(Tscal gamma) { eos_gamma = gamma; }

        bool use_consistent_transport = false;
        bool use_van_leer             = true;
    };

} // namespace shammodels::zeus
