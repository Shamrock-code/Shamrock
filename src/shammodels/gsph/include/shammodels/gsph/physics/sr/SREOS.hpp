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
 * @file SREOS.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Equation of State computation for Special Relativistic GSPH
 *
 * Computes pressure and sound speed with relativistic corrections.
 * Based on Kitajima et al. (2025) arXiv:2510.18251v1
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief EOS computation module for Special Relativistic physics
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class SREOS {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        /**
         * @brief Compute pressure and sound speed fields
         *
         * For SR adiabatic EOS: P = (gamma - 1) * n * epsilon
         * where n is rest-frame density, epsilon is specific internal energy
         * Sound speed includes relativistic corrections: cs = sqrt((gamma-1)(H-1)/H) * c
         */
        static void compute(Storage &storage, const Config &config, PatchScheduler &scheduler);

        /**
         * @brief Compute output density field for I/O
         *
         * SR: Converts lab-frame N to rest-frame n = N/gamma
         */
        static void compute_output_density_field(
            PatchScheduler &scheduler,
            const Config &config,
            Tscal c_speed,
            shamrock::ComputeField<Tscal> &density);

        /**
         * @brief Compute output pressure field for I/O
         */
        static void compute_output_pressure_field(
            PatchScheduler &scheduler,
            const Config &config,
            Tscal c_speed,
            shamrock::ComputeField<Tscal> &pressure);
    };

} // namespace shammodels::gsph::physics::sr
