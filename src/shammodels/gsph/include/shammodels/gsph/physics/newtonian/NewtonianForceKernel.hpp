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
 * @file NewtonianForceKernel.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Force kernel implementation for Newtonian GSPH
 *
 * Implements the GSPH force computation for Newtonian hydrodynamics.
 * Supports multiple Riemann solvers (Iterative, HLL, HLLC, Roe).
 *
 * The force computation follows Cha & Whitworth (2003):
 * dv_i/dt = -Σ_j m_j p* [V_i² ∇W_ij(h_i) + V_j² ∇W_ij(h_j)]
 *
 * Where p* is computed from the 1D Riemann problem along the pair axis.
 */

#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shammodels/gsph/physics/newtonian/riemann/RiemannBase.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::newtonian {

    using ::PatchScheduler;

    /**
     * @brief Force kernel for Newtonian GSPH
     *
     * Computes accelerations and energy rates using GSPH with configurable
     * Riemann solvers. Each Riemann solver type has its own compute method.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class NewtonianForceKernel {
        public:
        using Tscal   = shambase::VecComponent<Tvec>;
        using Storage = SolverStorage<Tvec, u32>;
        using Config  = SolverConfig<Tvec, SPHKernel>;

        NewtonianForceKernel(PatchScheduler &scheduler, const Config &config, Storage &storage)
            : scheduler_(scheduler), config_(config), storage_(storage) {}

        /**
         * @brief Compute forces using iterative Riemann solver (van Leer 1997)
         * @param cfg Configuration for iterative solver
         */
        void compute_iterative(const riemann::IterativeConfig &cfg);

        /**
         * @brief Compute forces using HLL approximate Riemann solver
         * @param cfg Configuration for HLL solver
         */
        void compute_hll(const riemann::HLLConfig &cfg);

        /**
         * @brief Compute forces using HLLC approximate Riemann solver
         * @param cfg Configuration for HLLC solver
         */
        void compute_hllc(const riemann::HLLCConfig &cfg);

        /**
         * @brief Compute forces using Roe linearized Riemann solver
         * @param cfg Configuration for Roe solver
         */
        void compute_roe(const riemann::RoeConfig &cfg);

        private:
        PatchScheduler &scheduler_;
        const Config &config_;
        Storage &storage_;
    };

} // namespace shammodels::gsph::physics::newtonian
