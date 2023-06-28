// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambase/type_aliases.hpp"
#include "shammodels/sph/SPHModelSolverConfig.hpp"
#include "shammodels/sph/modules/SPHSolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class UpdateViscosity {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SPHModelSolverConfig<Tvec, SPHKernel>;
        using Storage = SPHSolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        UpdateViscosity(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void update_artificial_viscosity(Tscal dt);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        void update_artificial_viscosity_mm97(Tscal dt, typename Config::AVConfig::VaryingMM97 cfg);
        void update_artificial_viscosity_cd10(Tscal dt, typename Config::AVConfig::VaryingCD10 cfg);
    };

} // namespace shammodels::sph::modules