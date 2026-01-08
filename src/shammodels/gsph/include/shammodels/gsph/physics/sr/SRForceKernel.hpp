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
 * @file SRForceKernel.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Force kernel implementation for Special Relativistic GSPH
 *
 * Implements the GSPH force computation for SR hydrodynamics following
 * Kitajima, Inutsuka, and Seno (2025) - arXiv:2510.18251v1
 *
 * Key differences from Newtonian:
 * - Uses exact relativistic Riemann solver (Pons et al. 2000)
 * - Includes tangent velocity preservation
 * - Computes dS/dt and de/dt instead of dv/dt and du/dt
 * - Uses lab-frame volume V = ν/N (not rest-frame)
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shammodels/gsph/physics/sr/riemann/RiemannBase.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::sr {

    using ::PatchScheduler;

    /**
     * @brief Force kernel for Special Relativistic GSPH
     *
     * Computes momentum derivatives (dS/dt) and energy derivatives (de/dt)
     * using the exact SR Riemann solver.
     *
     * The force computation follows Kitajima et al. (2025):
     * dS_i/dt = -Σ_j ν_j P* [V_i² ∇W_ij(h_i) + V_j² ∇W_ij(h_j)]
     *
     * Where P* is computed from the exact relativistic Riemann problem.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, TGauss3)
     */
    template<class Tvec, template<class> class SPHKernel>
    class SRForceKernel {
        public:
        using Tscal   = shambase::VecComponent<Tvec>;
        using Storage = SolverStorage<Tvec, u32>;
        using Config  = SolverConfig<Tvec, SPHKernel>;

        SRForceKernel(PatchScheduler &scheduler, const Config &config, Storage &storage)
            : scheduler_(scheduler), config_(config), storage_(storage) {}

        /**
         * @brief Compute forces using exact SR Riemann solver
         *
         * Computes dS/dt and de/dt for all particles using the exact
         * relativistic Riemann solver of Pons et al. (2000).
         *
         * @param cfg Configuration for exact solver (tolerance, max iterations)
         */
        void compute_exact(const riemann::ExactConfig &cfg);

        /**
         * @brief Compute forces using HLLC-SR approximate solver
         *
         * Faster but less accurate than exact solver.
         *
         * @param cfg Configuration for HLLC-SR solver
         */
        void compute_hllc(const riemann::HLLC_SRConfig &cfg);

        private:
        PatchScheduler &scheduler_;
        const Config &config_;
        Storage &storage_;
    };

} // namespace shammodels::gsph::physics::sr
