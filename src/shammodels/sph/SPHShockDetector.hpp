// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/stacktrace.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/SPHModelSolverConfig.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels {

    template<class Tvec, template<class> class SPHKernel>
    class SPHShockDetector {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SPHModelSolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
        SPHShockDetector(ShamrockCtx &context) : context(context) {}

        void update_artificial_viscosity_mm97(Tscal dt, typename Config::AVConfig::VaryingMM97 cfg);
        void update_artificial_viscosity_cd10(Tscal dt, typename Config::AVConfig::VaryingCD10 cfg);

        void update_artificial_viscosity(Tscal dt, typename Config::AVConfig::Variant cfg);
    };

} // namespace shammodels
