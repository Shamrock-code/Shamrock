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

    struct OptIndexList {
        std::optional<sycl::buffer<u32>> idx;
        u32 count;
    };

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

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        AMRGridRefinementHandler(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void update_refinement();

        private:
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
            shambase::DistributedData<sycl::buffer<u32>> &refine_flags,
            shambase::DistributedData<sycl::buffer<u32>> &derefine_flags,
            T &&...args);

        /**
         * @brief Enforces the 2:1 refinement ratio for blocks.
         *
         * This function iterates through blocks marked for refinement and ensures that
         * adjacent, coarser blocks are also marked for refinement to maintain the 2:1
         * grid balance. This is done iteratively to propagate the refinement as needed.
         * @param refine_flags refinement flags
         * @param refine_list        refinement maps
         */
        void enforce_two_to_one_for_refinement(
            shambase::DistributedData<sycl::buffer<u32>> &&refine_flags,
            shambase::DistributedData<OptIndexList> &refine_list);

        /**
         * @brief Check geometrical validity for derefinement
         *
         * This function iterates over all blocks flagged for derefinement and checks
         * whether all of their siblings also request derefinement, and if the merge operation can
         * be done. If these conditions are satisfied, the merge (coarsening) operation is
         * considered valid.
         *
         * To avoid duplicate operations, only the first block among each group of
         * eight siblings is retained when all validity checks succeed.
         *
         * @param derefine_flags  Derefinement flags
         * @param refine_flags    Refinement flags
         */
        void check_geometrical_validity_for_derefinement(
            shambase::DistributedData<sycl::buffer<u32>> &&derefine_flags,
            shambase::DistributedData<sycl::buffer<u32>> &&refine_flags);

        template<class UserAcc>
        bool internal_refine_grid(shambase::DistributedData<OptIndexList> &&refine_list);

        template<class UserAcc>
        bool internal_derefine_grid(shambase::DistributedData<OptIndexList> &&derefine_list);

        template<class UserAccCrit, class UserAccSplit, class UserAccMerge>
        void internal_update_refinement();

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
