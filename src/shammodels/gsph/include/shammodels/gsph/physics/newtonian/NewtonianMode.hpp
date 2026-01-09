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
 * @file NewtonianMode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Newtonian (non-relativistic) physics mode for GSPH
 *
 * Implements standard GSPH with leapfrog (kick-drift-kick) time stepping.
 * Works directly with primitive variables (v, u) - no conserved formulation.
 *
 * Timestep sequence:
 *   predictor → boundary → tree → omega → gradients → eos → forces → corrector
 */

#include "shammodels/gsph/core/PhysicsMode.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianFieldNames.hpp"
#include "shammodels/gsph/physics/newtonian/riemann/RiemannBase.hpp"

namespace shammodels::gsph::physics::newtonian {

    /**
     * @brief Newtonian physics mode implementation
     *
     * Owns the complete timestep - no query methods needed.
     */
    template<class Tvec, template<class> class SPHKernel>
    class NewtonianMode : public core::PhysicsMode<Tvec, SPHKernel> {
        public:
        using Base    = core::PhysicsMode<Tvec, SPHKernel>;
        using Tscal   = typename Base::Tscal;
        using Storage = typename Base::Storage;
        using Config  = typename Base::Config;

        NewtonianMode() = default;

        // ════════════════════════════════════════════════════════════════════════
        // Core Interface
        // ════════════════════════════════════════════════════════════════════════

        Tscal evolve_timestep(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) override;

        void init_fields(Storage &storage, Config &config) override;

        // ════════════════════════════════════════════════════════════════════════
        // Metadata
        // ════════════════════════════════════════════════════════════════════════

        std::string_view name() const override { return "Newtonian"; }

        std::string_view description() const override {
            return "Newtonian GSPH with leapfrog integration";
        }

        const char *get_density_field_name() const override { return fields::DENSITY; }

        // ════════════════════════════════════════════════════════════════════════
        // Layout Extension - Newtonian-specific fields
        // ════════════════════════════════════════════════════════════════════════

        void extend_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout) override {
            // Mass density field ρ (Newtonian's fundamental density)
            ghost_layout.add_field<Tscal>(fields::DENSITY, 1);
        }

        private:
        // ════════════════════════════════════════════════════════════════════════
        // Internal Implementation
        // ════════════════════════════════════════════════════════════════════════

        void do_predictor(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);
        bool apply_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);
        void prepare_corrector(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void compute_forces(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void compute_eos(Storage &storage, const Config &config, PatchScheduler &scheduler);

        /**
         * @brief Newtonian-specific h-iteration and density computation
         *
         * Uses standard mass-based SPH:
         *   ρ = Σ m W(r_ij, h_i)      - Mass density
         *   h chosen so ρ_sum = ρ_h   - Standard SPH h-iteration (no c_smooth)
         *
         * @return true if h converged, false if needs more subcycles
         */
        bool compute_omega_newtonian(
            Storage &storage, const Config &config, PatchScheduler &scheduler);

        void compute_forces_iterative(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const riemann::IterativeConfig &riemann_config);

        void compute_forces_hll(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const riemann::HLLConfig &riemann_config);

        void compute_forces_hllc(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const riemann::HLLCConfig &riemann_config);
    };

} // namespace shammodels::gsph::physics::newtonian
