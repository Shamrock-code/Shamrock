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
 * @file AMRGridRefinementHandler.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/amr/AMRCell.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class AMRGridRefinementHandler {

        class AMRBlockFinder;
        class AMRLowering;

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton         = u64;
        using AMRBlock         = typename Config::AMRBlock;
        using BlockCoord       = shamrock::amr::AMRBlockCoord<TgridVec, 3>;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        AMRGridRefinementHandler(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void update_refinement();

        private:
        /**
         * @brief build histrogram for amr block levels
         */
        void amr_block_levels_histogram(
            const sham::DeviceScheduler_ptr &sched,
            const sham::DeviceBuffer<TgridUint> &amr_levels,
            const u32 nb_levels,
            sham::DeviceBuffer<u32> &block_count_per_level,
            sham::DeviceBuffer<u32> &block_reordered_indx_map,
            const u32 len,
            TgridUint level_min,
            TgridUint level_max);

        /**
         * @brief Generate the list of blocks that need to be refined or derefined.
         *
         * We then need to apply the refinement, apply the changes to the indexes in the derefine
         * list, then apply the derefinement.
         *
         * @tparam UserAcc
         * @tparam Fct
         * @tparam T
         * @param refine_list
         * @param derefine_list
         * @param flag_refine_derefine_functor
         * @param args
         */
        template<class UserAcc, class... T>
        void gen_refine_block_changes(
            shambase::DistributedData<sham::DeviceBuffer<u32>> &refine_list,
            shambase::DistributedData<sham::DeviceBuffer<u32>> &derefine_list,
            T &&...args);

        template<class UserAcc>
        bool internal_refine_grid(shambase::DistributedData<sham::DeviceBuffer<u32>> &&refine_list);

        template<class UserAcc>
        bool internal_derefine_grid(
            shambase::DistributedData<sham::DeviceBuffer<u32>> &&derefine_list);

        template<class UserAccCrit, class UserAccSplit, class UserAccMerge>
        void internal_update_refinement();

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
