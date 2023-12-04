// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file UpdateDerivs.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class UpdateDerivs {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        UpdateDerivs(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void update_derivs();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        using Cfg_AV = typename Config::AVConfig;

        using None        = typename Cfg_AV::None;
        using Constant    = typename Cfg_AV::Constant;
        using VaryingMM97 = typename Cfg_AV::VaryingMM97;
        using VaryingCD10 = typename Cfg_AV::VaryingCD10;
        using ConstantDisc = typename Cfg_AV::ConstantDisc;

        void update_derivs_noAV(None cfg);
        void update_derivs_constantAV(Constant cfg);
        void update_derivs_mm97(VaryingMM97 cfg);
        void update_derivs_cd10(VaryingCD10 cfg);
        void update_derivs_disc_visco(ConstantDisc cfg);
    };


} // namespace shammodels::sph::modules