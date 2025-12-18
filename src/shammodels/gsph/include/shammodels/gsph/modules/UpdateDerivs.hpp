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
 * @file UpdateDerivs.hpp
 * @author Guo
 * @brief GSPH derivative update module
 *
 * This module computes the time derivatives (acceleration and energy rate)
 * using the Godunov SPH formulation with Riemann solvers.
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 *
 * This implementation follows:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of
 *   Godunov-type particle hydrodynamics"
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::gsph::modules {

    /**
     * @brief GSPH derivative update module
     *
     * Computes the acceleration (dv/dt) and energy rate (du/dt) for all particles
     * using the GSPH formulation. For each particle pair:
     * 1. Extract left/right states (rho, v, P)
     * 2. Solve 1D Riemann problem along pair axis to get (p*, v*)
     * 3. Compute force contribution using p* instead of artificial viscosity
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4, M6, C2, C4, C6 for Wendland)
     */
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

        /**
         * @brief Update all derivatives using GSPH Riemann solver approach
         *
         * Dispatches to the appropriate implementation based on RiemannConfig.
         */
        void update_derivs();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }

        using Cfg_Riemann = typename Config::RiemannConfig;
        using Iterative   = typename Cfg_Riemann::Iterative;
        using Exact       = typename Cfg_Riemann::Exact;
        using HLLC        = typename Cfg_Riemann::HLLC;
        using Roe         = typename Cfg_Riemann::Roe;

        /**
         * @brief Update derivatives using iterative Riemann solver (van Leer 1997)
         */
        void update_derivs_iterative(Iterative cfg);

        /**
         * @brief Update derivatives using HLLC approximate Riemann solver
         */
        void update_derivs_hllc(HLLC cfg);
    };

} // namespace shammodels::gsph::modules
