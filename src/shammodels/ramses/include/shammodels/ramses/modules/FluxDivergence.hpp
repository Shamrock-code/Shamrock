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
 * @file FluxDivergence.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Leodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/ComputeFlux.hpp"
#include "shammodels/ramses/modules/ComputeGradient.hpp"
#include "shammodels/ramses/modules/ComputeTimeDerivative.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/FaceInterpolate.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class FluxDivergence {

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

        FluxDivergence(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /** compute the Flux at cell interfaces by solving Riemann problems*/
        // modules::ComputeFlux flux_compute(context, solver_config, storage);

        void eval_flux_divergence_hydro_fields();
        void eval_flux_divergence_dust_fields();

        void reset_storage_buffers_hydro();
        void reset_storage_buffers_dust();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
