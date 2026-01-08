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
 * @file BuildTrees.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief GSPH spatial tree building for neighbor search
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class BuildTrees {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;
        using RTree   = typename Config::RTree;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        BuildTrees(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void build_merged_pos_trees();
        void compute_presteps_rint();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::gsph::modules
