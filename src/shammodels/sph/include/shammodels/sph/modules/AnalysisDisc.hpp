// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AnalysisDisc.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
#include "shamphys/SodTube.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisDisc {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        u32 Nbin = 300;

        struct analysis_val {
            int ibin;
            Tscal radius;
            int counter;
            Tvec J;
            Tscal Sigma;
            Tvec l;
            Tscal tilt;
            Tscal twist;
            Tscal psi;
            Tscal H_on_r;
        };

        struct analysis_basis {

            sham::DeviceBuffer<Tscal> radius;
            sham::DeviceBuffer<Tscal> lx;
            sham::DeviceBuffer<Tscal> ly;
            sham::DeviceBuffer<Tscal> lz;
            sham::DeviceBuffer<Tscal> Sigma;
            sham::DeviceBuffer<Tscal> zmean;
        };

        struct analysis_stage1 {
            sham::DeviceBuffer<Tscal> tilt;
            sham::DeviceBuffer<Tscal> twist;
            sham::DeviceBuffer<Tscal> psi;
        };

        struct analysis_stage2 {

            sycl::buffer<Tscal> H;
            sycl::buffer<Tscal> H_on_R;
        };

        analysis_val compute_analysis();
        analysis_stage1 compute_analysis_stage1(analysis_basis &basis);
        analysis_stage2 compute_analysis_stage2(analysis_stage1 &stage1);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
