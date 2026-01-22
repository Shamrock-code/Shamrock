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
 * @file VTKDump.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief VTK dump module for GSPH solver
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class VTKDump {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        Config &solver_config;

        VTKDump(ShamrockCtx &context, Config &solver_config)
            : context(context), solver_config(solver_config) {}

        void do_dump(std::string filename, bool add_patch_world_id);

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::gsph::modules
