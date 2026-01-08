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
 * @file SRMode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Special Relativistic physics mode for GSPH
 *
 * Implements SR-GSPH with conserved variable integration:
 * - Conserved: S = γHv (momentum), e = γH - P/(Nc²) (energy)
 *
 * Timestep sequence (differs from Newtonian):
 *   predictor → boundary → tree → omega → gradients → eos → forces
 *   → primitive_recovery → corrector
 *
 * Based on Kitajima, Inutsuka, Seno (2025) arXiv:2510.18251v1
 */

#include "shammodels/gsph/core/PhysicsMode.hpp"
#include "shammodels/gsph/physics/sr/SRConfig.hpp"
#include "shammodels/gsph/physics/sr/SRFieldNames.hpp"

namespace shammodels::gsph::physics::sr {

    /**
     * @brief Special Relativistic physics mode implementation
     *
     * Owns the complete timestep including SR-specific primitive recovery.
     */
    template<class Tvec, template<class> class SPHKernel>
    class SRMode : public core::PhysicsMode<Tvec, SPHKernel> {
        public:
        using Base    = core::PhysicsMode<Tvec, SPHKernel>;
        using Tscal   = typename Base::Tscal;
        using Storage = typename Base::Storage;
        using Config  = typename Base::Config;
        using SRCfg   = SRConfig<Tvec>;

        SRMode() = default;
        explicit SRMode(const SRCfg &sr_config) : sr_config_(sr_config) {}

        // ════════════════════════════════════════════════════════════════════════
        // Core Interface
        // ════════════════════════════════════════════════════════════════════════

        Tscal evolve_timestep(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            Tscal dt,
            const core::SolverCallbacks<Tscal> &callbacks) override;

        void init_fields(Storage &storage, Config &config) override;

        // ════════════════════════════════════════════════════════════════════════
        // Metadata
        // ════════════════════════════════════════════════════════════════════════

        std::string_view name() const override { return "SpecialRelativistic"; }

        std::string_view description() const override {
            return "SR-GSPH with conserved variable integration (Kitajima et al. 2025)";
        }

        std::vector<std::string> get_output_field_names() const override {
            // SR outputs rest-frame density (stored directly, no transformation)
            return {"density", "pressure", "velocity", "soundspeed", "lorentz_factor"};
        }

        const char *get_density_field_name() const override { return fields::N_LABFRAME; }

        // ════════════════════════════════════════════════════════════════════════
        // Layout Extension - SR-specific fields
        // ════════════════════════════════════════════════════════════════════════

        void extend_layout(shamrock::patch::PatchDataLayerLayout &pdl) override;
        void extend_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout) override;

        bool uses_pmass_field() const override { return true; }

        private:
        // SR-specific compute_omega with volume-based h
        bool compute_omega_sr(Storage &storage, const Config &config, PatchScheduler &scheduler);

        // ════════════════════════════════════════════════════════════════════════
        // Internal Implementation
        // ════════════════════════════════════════════════════════════════════════

        SRCfg sr_config_{}; ///< SR-specific configuration
        bool sr_initialized_       = false;
        bool first_real_step_done_ = false; ///< Track if first step with dt>0 completed

        void init_conserved(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void clear_derivatives(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void do_predictor(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);
        bool apply_corrector(
            Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt);
        void prepare_corrector(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void compute_forces(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void compute_eos(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void recover_primitives(Storage &storage, const Config &config, PatchScheduler &scheduler);
        void update_acceleration_for_cfl(
            Storage &storage, const Config &config, PatchScheduler &scheduler);
        void check_derivatives_for_nan(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            const std::string &context);
    };

} // namespace shammodels::gsph::physics::sr
