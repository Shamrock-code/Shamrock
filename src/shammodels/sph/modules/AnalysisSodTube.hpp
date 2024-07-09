// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ComputeEos.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/vec.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamphys/SodTube.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisSodTube {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        shamphys::SodTube solution;

        AnalysisSodTube(ShamrockCtx &context, Config &solver_config, Storage &storage, shamphys::SodTube& solution)
            : context(context), solver_config(solver_config), storage(storage), solution(solution) {}

        struct field_val {
            Tscal rho;
            Tvec v;
            Tscal P;
        };

        field_val compute_L2_dist();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };


} // namespace shammodels::sph::modules