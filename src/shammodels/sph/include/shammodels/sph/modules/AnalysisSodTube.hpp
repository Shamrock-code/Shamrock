// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AnalysisSodTube.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
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
        Tvec direction;
        Tscal time_val;
        Tscal x_ref;        // shock centered on x_ref
        Tscal x_min, x_max; // check only between [x_min, x_max ]

        AnalysisSodTube(
            ShamrockCtx &context,
            Config &solver_config,
            Storage &storage,
            shamphys::SodTube &solution,
            Tvec direction,
            Tscal time_val,
            Tscal x_ref,
            Tscal x_min,
            Tscal x_max)
            : context(context), solver_config(solver_config), storage(storage), solution(solution),
              direction(direction), time_val(time_val), x_ref(x_ref), x_min(x_min), x_max(x_max) {}

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
