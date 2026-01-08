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
 * @file NewtonianTimestepper.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Leapfrog time integration for Newtonian GSPH
 *
 * Implements kick-drift-kick (KDK) leapfrog scheme:
 * - Predictor: v += a*dt/2; x += v*dt; u += du*dt/2
 * - Corrector: v += a*dt/2; u += du*dt/2
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::newtonian {

    /**
     * @brief Leapfrog time integration for Newtonian physics
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class NewtonianTimestepper {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        /**
         * @brief Predictor half-step (first kick + drift)
         *
         * v += a*dt/2 (first kick)
         * x += v*dt   (drift)
         * u += du*dt/2 (if adiabatic)
         */
        static void do_predictor(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);

        /**
         * @brief Corrector half-step (second kick)
         *
         * v += a*dt/2 (second kick with NEW acceleration)
         * u += du*dt/2 (if adiabatic)
         *
         * @return true always (Newtonian has no failure modes)
         */
        static bool apply_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);

        /**
         * @brief Save old derivatives before force computation
         *
         * Stores old_axyz and old_duint for averaging in corrector step.
         */
        static void prepare_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler);
    };

} // namespace shammodels::gsph::physics::newtonian
