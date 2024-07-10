// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file UpdateDustDerivsTVI.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambackends/vec.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class UpdateDustDerivsTVI {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        UpdateDustDerivsTVI(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void update_dust_derivs();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        using DustCFG     = typename Config::DustConfig;
        using DustCFGNone = typename DustCFG::None;
        using DustCFGMonofluidTVI = typename DustCFG::DustMonofluidTvi;

        void update_dust_derivs_tvi(DustCFGMonofluidTVI config);

    };

} // namespace shammodels::sph::modules