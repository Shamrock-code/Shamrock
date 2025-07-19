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

#include "shamalgs/details/numeric/numeric.hpp"
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

        u32 Nbin   = 300;
        Tscal Rmin = 0.1;
        Tscal Rmax = 1.0;

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
            sham::DeviceBuffer<u64> counter;
            sham::DeviceBuffer<Tscal> Jx;
            sham::DeviceBuffer<Tscal> Jy;
            sham::DeviceBuffer<Tscal> Jz;
            sham::DeviceBuffer<Tscal> zmean;
            sham::DeviceBuffer<Tscal> Sigma;
        };

        struct analysis_stage0 {
            sham::DeviceBuffer<Tscal> lx;
            sham::DeviceBuffer<Tscal> ly;
            sham::DeviceBuffer<Tscal> lz;
        };

        struct analysis_stage1 {
            sham::DeviceBuffer<Tscal> tilt;
            sham::DeviceBuffer<Tscal> twist;
            sham::DeviceBuffer<Tscal> psi;
        };

        struct analysis_stage2 {

            sham::DeviceBuffer<Tscal> H;
            sham::DeviceBuffer<Tscal> H_on_R; // @@@ yes this is redundant, let's keep it for now
        };

        analysis_val compute_analysis();
        analysis_basis compute_analysis_basis(
            Tscal Rmin, Tscal Rmax, u32 Nbin, const sham::DeviceScheduler_ptr &sched);
        analysis_stage0 compute_analysis_stage0(analysis_basis &basis);
        analysis_stage1 compute_analysis_stage1(analysis_basis &basis, analysis_stage0 &stage0);
        analysis_stage2 compute_analysis_stage2(analysis_stage1 &stage1);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
