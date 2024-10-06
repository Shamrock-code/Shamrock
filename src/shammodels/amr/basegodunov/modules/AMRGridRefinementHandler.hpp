// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AMRGridRefinementHandler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/amr/NeighGraph.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
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

        void refine_grid(shambase::DistributedData<OptIndexList> refine_list);
        void derefine_grid(shambase::DistributedData<OptIndexList> derefine_list);

        private:
        template<class UserAcc, class Fct>
        void internal_refine_grid(
            shambase::DistributedData<OptIndexList> &&refine_list, Fct &&refine_functor);

        template<class UserAcc, class Fct>
        void internal_derefine_grid(
            shambase::DistributedData<OptIndexList> &&derefine_list, Fct &&derefine_functor);

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
