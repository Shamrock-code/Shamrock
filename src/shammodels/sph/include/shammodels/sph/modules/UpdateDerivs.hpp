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
 * @file UpdateDerivs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
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

        using None         = typename Cfg_AV::None;
        using Constant     = typename Cfg_AV::Constant;
        using VaryingMM97  = typename Cfg_AV::VaryingMM97;
        using VaryingCD10  = typename Cfg_AV::VaryingCD10;
        using ConstantDisc = typename Cfg_AV::ConstantDisc;

        void update_derivs_noAV(None cfg);
        void update_derivs_constantAV(Constant cfg);
        void update_derivs_mm97(VaryingMM97 cfg);
        void update_derivs_cd10(VaryingCD10 cfg);
        void update_derivs_disc_visco(ConstantDisc cfg);

        using Cfg_MHD = typename Config::MHDConfig;

        using NoneMHD     = typename Cfg_MHD::None;
        using IdealMHD    = typename Cfg_MHD::IdealMHD_constrained_hyper_para;
        using NonIdealMHD = typename Cfg_MHD::NonIdealMHD;

        void update_derivs_MHD(IdealMHD cfg);

        using Cfg_Dust = typename Config::DustConfig;

        using NoDust             = typename Cfg_Dust::None;
        using FullOneFluidConfig = typename Cfg_Dust::MonofluidComplete;
        void
        update_derivs_cd10_full_one_fluid(VaryingCD10 cfg_visco, FullOneFluidConfig cfg_one_fluid);
    };

} // namespace shammodels::sph::modules
