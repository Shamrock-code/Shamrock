// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file BarotropicEnergyReset.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Internal energy reset onto the barotrope of the ResetBarotropic EOS
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"

namespace shammodels::basegodunov::modules {

    /**
     * @brief Overwrite the gas internal energy by the barotropic one (EOSConfig::ResetBarotropic)
     *
     * For every cell, in place in the patch data, keeping rho and rhovel:
     * \f[
     *   \rho e_{\rm tot} \leftarrow \frac{P_{\rm baro}(\rho)}{\gamma - 1}
     *       + \frac{1}{2} \frac{(\rho \mathbf{v})^2}{\rho}
     * \f]
     * with \f$ \gamma \f$ the adiabatic index of the hydro step.
     */
    template<class Tvec, class TgridVec>
    class BarotropicEnergyReset {

        public:
        using Tscal = shambase::VecComponent<Tvec>;

        using Config   = SolverConfig<Tvec, TgridVec>;
        using Storage  = SolverStorage<Tvec, TgridVec, u64>;
        using AMRBlock = typename Config::AMRBlock;

        ShamrockCtx &context;
        Config &solver_config;
        Storage &storage;

        BarotropicEnergyReset(ShamrockCtx &context, Config &solver_config, Storage &storage)
            : context(context), solver_config(solver_config), storage(storage) {}

        /// Apply the reset, does nothing if the EOS is not EOSConfig::ResetBarotropic
        void reset_energy();

        private:
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
    };

} // namespace shammodels::basegodunov::modules
