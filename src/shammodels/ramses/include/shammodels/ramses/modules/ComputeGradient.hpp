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
    class ComputeGradient {

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

        ComputeGradient(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        void compute_grad_rho_van_leer();
        void compute_grad_v_van_leer();
        void compute_grad_P_van_leer();
        void compute_grad_rho_dust_van_leer();
        void compute_grad_v_dust_van_leer();

        private:
        template<SlopeMode mode>
        void _compute_grad_rho_van_leer();
        template<SlopeMode mode>
        void _compute_grad_v_van_leer();
        template<SlopeMode mode>
        void _compute_grad_P_van_leer();
        template<SlopeMode mode>
        void _compute_grad_rho_dust_van_leer();
        template<SlopeMode mode>
        void _compute_grad_v_dust_van_leer();

        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
