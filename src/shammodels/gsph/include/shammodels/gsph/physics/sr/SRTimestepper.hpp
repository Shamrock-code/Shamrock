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
 * @file SRTimestepper.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Time integration for Special Relativistic GSPH
 *
 * Implements conserved variable integration with position drift.
 * Based on Kitajima et al. (2025) arXiv:2510.18251v1
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief Time integration for Special Relativistic physics
     *
     * Uses conserved variables (S, e) with half-step updates.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class SRTimestepper {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        /**
         * @brief Predictor half-step
         *
         * S += dS * dt/2, e += de * dt/2
         * Then recover primitives and drift: x += v*dt
         */
        static void do_predictor(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);

        /**
         * @brief Corrector half-step
         *
         * S += dS * dt/2, e += de * dt/2
         * Then recover primitives
         *
         * @return true always (could check for superluminal particles)
         */
        static bool apply_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);

        /**
         * @brief Prepare for corrector (no-op for SR)
         *
         * SR doesn't need to save old derivatives - predictor-corrector is symmetric.
         */
        static void prepare_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler);
    };

} // namespace shammodels::gsph::physics::sr
