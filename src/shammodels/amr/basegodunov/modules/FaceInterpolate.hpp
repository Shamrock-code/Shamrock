// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file FaceInterpolate.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"

#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"
#include "shammodels/amr/NeighGraph.hpp"
#include "shamrock/scheduler/ComputeField.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class FaceInterpolate {

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton = u64;
        using AMRBlock         = typename Config::AMRBlock;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        FaceInterpolate(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void interpolate_rho_to_face();
        void interpolate_v_to_face();
        void interpolate_P_to_face();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

    };

} // namespace shammodels::basegodunov::modules
