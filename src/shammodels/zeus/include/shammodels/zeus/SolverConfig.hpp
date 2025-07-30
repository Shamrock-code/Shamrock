// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"
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

        inline void check_config() {
            if (grid_coord_to_pos_fact <= 0) {
                shambase::throw_with_loc<std::runtime_error>(shambase::format(
                    "grid_coord_to_pos_fact must be > 0, got {}", grid_coord_to_pos_fact));
            }
        }
    };

} // namespace shammodels::zeus
