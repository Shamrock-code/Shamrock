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
 * @file DiffOperator.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/zeus/Solver.hpp"
#include "shammodels/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    template<class Tvec, class TgridVec>
    class DiffOperator {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        DiffOperator(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief compute gradient tensor of the velocity from existing data
         *
         */
        void compute_gradu();

        /**
         * @brief exchange ghost gradu
         *
         */
        void exchange_gradu();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::zeus::modules
