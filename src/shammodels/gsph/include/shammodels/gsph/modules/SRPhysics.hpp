// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SRPhysics.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Special Relativistic GSPH physics module
 *
 * This module handles the SR-specific integration steps:
 * - Initialization of conserved variables (S, e) from primitives
 * - Predictor step (half-step in conserved variables + position drift)
 * - Corrector step (second half-step + primitive recovery)
 * - Conservative to primitive variable recovery
 *
 * Based on Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 * "Special Relativistic Godunov SPH"
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::gsph::modules {

    /**
     * @brief Special Relativistic physics module for GSPH
     *
     * Manages the SR-specific aspects of the time integration:
     * - Uses conserved variables S (canonical momentum) and e (canonical energy)
     * - Integrates S and e forward using dS/dt and de/dt from UpdateDerivs
     * - Recovers primitive variables (v, P) from conserved variables
     *
     * Key equations:
     *   S = gamma * H * v  (canonical momentum)
     *   e = gamma * H - P / (N * c^2)  (canonical energy)
     *   H = 1 + u/c^2 + P/(n*c^2)  (specific enthalpy)
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4, M6)
     */
    template<class Tvec, template<class> class SPHKernel>
    class SRPhysics {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SRPhysics(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Initialize SR field storage (S, e, dS, de)
         *
         * Creates the storage fields for conserved variables and their derivatives
         * if they don't already exist.
         */
        void init_fields();

        /**
         * @brief Initialize conserved variables from primitives
         *
         * Computes S = gamma * H * v and e = gamma * H - P/(N*c^2)
         * from the current velocity, density, and pressure fields.
         *
         * Should be called once at the start of SR simulation before first step.
         */
        void init_conserved();

        /**
         * @brief SR predictor step
         *
         * 1. Half-step: S += dS * dt/2, e += de * dt/2
         * 2. Primitive recovery: v, P from S, e
         * 3. Position drift: x += v * dt
         *
         * @param dt Timestep
         */
        void do_predictor(Tscal dt);

        /**
         * @brief SR corrector step
         *
         * 1. Half-step: S += dS * dt/2, e += de * dt/2
         * 2. Primitive recovery: v, P from S, e
         *
         * @param dt Timestep
         */
        void apply_corrector(Tscal dt);

        /**
         * @brief Recover primitive variables from conserved
         *
         * Uses Newton-Raphson iteration to solve for the Lorentz factor,
         * then computes velocity and pressure from S, e, N.
         *
         * Key steps:
         * 1. Solve f(W) = 0 for Lorentz factor W = gamma
         * 2. v = S / (gamma * H)
         * 3. P = (gamma - 1) * n * (H - 1 - v^2*gamma/(gamma+1)) (approx)
         */
        void cons2prim();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::gsph::modules
