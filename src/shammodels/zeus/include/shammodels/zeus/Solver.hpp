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
 * @file Solver.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/zeus/SolverConfig.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::zeus {

    template<class Tvec, class TgridVec>
    class Solver {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using u_morton = u64;

        using Config = SolverConfig<Tvec, TgridVec>;

        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        Config solver_config;

        SolverStorage<Tvec, TgridVec, u_morton> storage{};

        inline void init_required_fields() {
            context.pdata_layout_add_field<TgridVec>("cell_min", 1);
            context.pdata_layout_add_field<TgridVec>("cell_max", 1);
            context.pdata_layout_add_field<Tscal>("rho", AMRBlock::block_size);
            context.pdata_layout_add_field<Tscal>("eint", AMRBlock::block_size);
            context.pdata_layout_add_field<Tvec>("vel", AMRBlock::block_size);
        }

        Solver(ShamrockCtx &context) : context(context) {}

        Tscal evolve_once(Tscal t_current, Tscal dt_input);
    };

} // namespace shammodels::zeus
