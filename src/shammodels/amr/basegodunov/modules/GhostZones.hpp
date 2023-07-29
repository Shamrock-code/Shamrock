// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/basegodunov/Solver.hpp"
#include "shammodels/amr/basegodunov/modules/SolverStorage.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class GhostZones{
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec>;
        using Storage = SolverStorage<Tvec, TgridVec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        GhostZones(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void build_ghost_cache();
        
        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

    };
}