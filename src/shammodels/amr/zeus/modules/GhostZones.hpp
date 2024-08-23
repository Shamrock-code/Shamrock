// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file GhostZones.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/SolverStorage.hpp"

namespace shammodels::zeus::modules {

    template<class Tvec, class TgridVec>
    class GhostZones {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, TgridVec>;
        using Storage = SolverStorage<Tvec, TgridVec, u64>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        GhostZones(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void build_ghost_cache();

        shambase::DistributedDataShared<shamrock::patch::PatchData> communicate_pdat(
            shamrock::patch::PatchDataLayout &pdl,
            shambase::DistributedDataShared<shamrock::patch::PatchData> &&interf);

        template<class T>
        shambase::DistributedDataShared<PatchDataField<T>>
        communicate_pdat_field(shambase::DistributedDataShared<PatchDataField<T>> &&interf);

        template<class T, class Tmerged>
        shambase::DistributedData<Tmerged> merge_native(
            shambase::DistributedDataShared<T> &&interfs,
            std::function<Tmerged(const shamrock::patch::Patch, shamrock::patch::PatchData &pdat)>
                init,
            std::function<void(Tmerged &, T &)> appender);

        void exchange_ghost();

        template<class T>
        shamrock::ComputeField<T> exchange_compute_field(shamrock::ComputeField<T> &in);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };
} // namespace shammodels::zeus::modules
