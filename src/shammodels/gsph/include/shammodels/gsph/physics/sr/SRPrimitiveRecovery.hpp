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
 * @file SRPrimitiveRecovery.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Primitive variable recovery for Special Relativistic GSPH
 *
 * Recovers primitive variables (v, P, u) from conserved (S, e, N).
 * Uses Newton-Raphson iteration to solve for Lorentz factor.
 * Based on Kitajima et al. (2025) arXiv:2510.18251v1
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief Config for Newton-Raphson primitive recovery
     */
    struct NewtonRaphsonConfig {
        f64 tol      = 1e-10;
        u32 max_iter = 100;
    };

    /**
     * @brief Config for Noble 2D primitive recovery
     */
    struct Noble2DConfig {
        f64 tol      = 1e-10;
        u32 max_iter = 100;
    };

    /**
     * @brief Primitive variable recovery for SR physics
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class SRPrimitiveRecovery {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        /**
         * @brief Initialize conserved variables from primitives
         *
         * Computes S = gamma*H*v and e = gamma*H - P/(N*c^2)
         * from initial (v, rho, P).
         */
        static void init_conserved(
            Storage &storage, const Config &config, PatchScheduler &scheduler);

        /**
         * @brief Recover primitives using default method
         *
         * Dispatches to Newton-Raphson by default.
         */
        static void recover(Storage &storage, const Config &config, PatchScheduler &scheduler);

        /**
         * @brief Recover primitives using Newton-Raphson iteration
         */
        static void recover_newton_raphson(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const NewtonRaphsonConfig &recovery_config);

        /**
         * @brief Recover primitives using Noble 2D method
         */
        static void recover_noble_2d(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const Noble2DConfig &recovery_config);
    };

} // namespace shammodels::gsph::physics::sr
