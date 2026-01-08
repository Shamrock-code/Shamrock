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
 * @file NewtonianEOS.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Equation of State computation for Newtonian GSPH
 *
 * Computes pressure and sound speed from density and internal energy
 * using either adiabatic (P = (gamma-1)*rho*u) or isothermal (P = cs^2*rho) EOS.
 */

#include "shambackends/typeAliasVec.hpp"
#include "shammodels/gsph/SolverConfig.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"

namespace shammodels::gsph::physics::newtonian {

    /**
     * @brief EOS computation module for Newtonian physics
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class NewtonianEOS {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Config  = SolverConfig<Tvec, SPHKernel>;
        using Storage = SolverStorage<Tvec, u32>;

        /**
         * @brief Compute pressure and sound speed fields
         *
         * For adiabatic EOS: P = (gamma - 1) * rho * u, cs = sqrt(gamma * P / rho)
         * For isothermal EOS: P = cs^2 * rho
         */
        static void compute(Storage &storage, const Config &config, PatchScheduler &scheduler);

        /**
         * @brief Compute output density field for I/O
         */
        static void compute_output_density_field(
            PatchScheduler &scheduler,
            const Config &config,
            shamrock::ComputeField<Tscal> &density);

        /**
         * @brief Compute output pressure field for I/O
         */
        static void compute_output_pressure_field(
            PatchScheduler &scheduler,
            const Config &config,
            shamrock::ComputeField<Tscal> &pressure);
    };

} // namespace shammodels::gsph::physics::newtonian
