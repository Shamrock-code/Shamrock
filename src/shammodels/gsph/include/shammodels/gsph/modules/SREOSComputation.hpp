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
 * @file SREOSComputation.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Special Relativistic EOS computation module for GSPH
 *
 * This module handles the SR-specific equation of state computation:
 * - Computes rest-frame density n = N/γ from lab-frame N
 * - Uses correct relativistic EOS: P = (γ_eos - 1) * n * ε
 * - Computes relativistic sound speed and enthalpy
 *
 * Separated from the main Solver to keep SR-specific logic isolated
 * and avoid runtime branching in the hot path.
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::gsph::modules {

    /**
     * @brief SR-GSPH EOS computation module
     *
     * Computes pressure and sound speed for special relativistic hydrodynamics.
     * Uses the relativistic EOS where rest-frame density n is used, not lab-frame N.
     *
     * Key formulas:
     *   n = N / γ                           (rest-frame density)
     *   P = (γ_eos - 1) * n * ε             (EOS with rest-frame density)
     *   H = 1 + ε/c² + P/(n·c²)             (specific enthalpy)
     *   cs² = (γ_eos - 1)(H - 1) / H        (relativistic sound speed)
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type
     */
    template<class Tvec, template<class> class SPHKernel>
    class SREOSComputation {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        SREOSComputation(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /**
         * @brief Compute SR EOS fields (pressure, sound speed)
         *
         * Reads: density (lab-frame N), uint (internal energy), vxyz (velocity)
         * Writes: pressure, soundspeed
         *
         * The key difference from Newtonian EOS:
         * - Uses n = N/γ (rest-frame density) for the EOS formula
         * - Computes relativistic sound speed using enthalpy
         */
        void compute_eos();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::gsph::modules
