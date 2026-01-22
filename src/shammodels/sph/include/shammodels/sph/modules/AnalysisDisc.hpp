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
 * @file AnalysisDisc.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/SolverStorage.hpp"
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

        AnalysisDisc(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /// TODO: make this a dichotomic search
        u32 mybin(Tscal radius, const Tscal *__restrict bin_edges, u32 Nbin) {
            u32 bini = 0;
            for (u32 bini = 0; bini < Nbin; bini++) {
                if (radius >= bin_edges[bini] && radius < bin_edges[bini + 1]) {
                    break;
                }
            }
            return bini;
        }

        struct analysis {
            sham::DeviceBuffer<Tscal> radius;
            sham::DeviceBuffer<u64> counter;
            sham::DeviceBuffer<Tscal> Sigma;
            sham::DeviceBuffer<Tscal> lx;
            sham::DeviceBuffer<Tscal> ly;
            sham::DeviceBuffer<Tscal> lz;
            sham::DeviceBuffer<Tscal> tilt;
            sham::DeviceBuffer<Tscal> twist;
            sham::DeviceBuffer<Tscal> psi;
            sham::DeviceBuffer<Tscal> Hsq;
        };

        struct analysis_basis {
            sham::DeviceBuffer<Tscal> buf_radius; // all radius for all particles
            sham::DeviceBuffer<Tscal> bin_edges;
            sham::DeviceBuffer<Tscal> radius; // binned radius
            sham::DeviceBuffer<u64> counter;
            sham::DeviceBuffer<Tscal> binned_Jx;
            sham::DeviceBuffer<Tscal> binned_Jy;
            sham::DeviceBuffer<Tscal> binned_Jz;
            sham::DeviceBuffer<Tscal> Sigma;
        };

        struct analysis_stage0 {
            sham::DeviceBuffer<Tscal> lx;
            sham::DeviceBuffer<Tscal> ly;
            sham::DeviceBuffer<Tscal> lz;
            sham::DeviceBuffer<Tscal> zmean;
            sham::DeviceBuffer<Tscal> Hsq;
        };

        struct analysis_stage1 {
            sham::DeviceBuffer<Tscal> tilt;
            sham::DeviceBuffer<Tscal> twist;
            sham::DeviceBuffer<Tscal> psi;
        };

        analysis_basis compute_analysis_basis(
            Tscal pmass, Tscal Rmin, Tscal Rmax, u32 Nbin, const ShamrockCtx &ctx);
        analysis_stage0 compute_analysis_stage0(analysis_basis &basis, u32 Nbin);
        analysis_stage1 compute_analysis_stage1(
            analysis_basis &basis, analysis_stage0 &stage0, u32 Nbin);

        analysis compute_analysis(Tscal Rmin, Tscal Rmax, u32 Nbin, const ShamrockCtx &ctx);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::sph::modules
