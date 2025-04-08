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
 * @file ComputeGradient.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Leodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class Slopes {

        public:
        using Tscal                      = shambase::VecComponent<Tvec>;
        using Tgridscal                  = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim         = shambase::VectorProperties<Tvec>::dimension;
        static constexpr u32 split_count = shambase::pow_constexpr<dim>(2);

        using Config           = SolverConfig<Tvec, TgridVec>;
        using Storage          = SolverStorage<Tvec, TgridVec, u64>;
        using u_morton         = u64;
        using AMRBlock         = typename Config::AMRBlock;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        Slopes(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void slope_rho();
        void slope_v();
        void slope_P();
        void slope_rho_dust();
        void slope_v_dust();

        private:
        template<SlopeMode mode>
        void _slope_rho();
        template<SlopeMode mode>
        void _slope_v();
        template<SlopeMode mode>
        void _slope_P();
        template<SlopeMode mode>
        void _slope_rho_dust();
        template<SlopeMode mode>
        void _slope_v_dust();

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
