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
 * @file PhysicsMode.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Minimal abstract interface for GSPH physics modes
 *
 * Each physics domain (Newtonian, SR, GR, MHD, Solid) implements this interface.
 * The interface is intentionally minimal to avoid combinatorial bloat.
 *
 * Design principles:
 * - PhysicsMode owns the entire timestep via evolve_timestep()
 * - No query methods (requires_X) - mode just does what it needs
 * - No output transformations - mode stores output-ready values
 * - Declarative I/O - mode declares which fields to export
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shammodels/gsph/modules/SolverStorage.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include <string_view>
#include <functional>
#include <string>
#include <vector>

// Forward declarations
namespace shammodels::gsph {
    template<class Tvec, template<class> class SPHKernel>
    struct SolverConfig;
} // namespace shammodels::gsph

namespace shammodels::gsph::core {

    using ::PatchScheduler;

    /**
     * @brief Callbacks for shared Solver operations
     *
     * Passed to PhysicsMode::evolve_timestep() to allow physics modes to
     * invoke shared operations (tree building, ghost handling, etc.)
     * without creating circular dependencies.
     */
    template<class Tscal>
    struct SolverCallbacks {
        // Tree and neighbor operations
        std::function<void()> gen_serial_patch_tree;
        std::function<void(Tscal)> gen_ghost_handler; ///< Takes time value
        std::function<void()> build_ghost_cache;
        std::function<void()> merge_position_ghost;
        std::function<void()> build_trees;
        std::function<void()> compute_presteps;
        std::function<void()> start_neighbors;
        std::function<bool()> compute_omega; ///< Returns h_converged

        // Ghost communication
        std::function<void()> init_ghost_layout;
        std::function<void()> communicate_ghosts;

        // Gradient computation
        std::function<void()> compute_gradients;

        // Density copy
        std::function<void()> copy_density;

        // CFL computation
        std::function<Tscal()> compute_cfl; ///< Returns dt_next

        // Cache reset for h-iteration
        std::function<void()> reset_for_h_iteration;

        // Cleanup
        std::function<void()> cleanup;

        // Config access
        u32 h_max_subcycles = 10;
    };

    /**
     * @brief Abstract base class for physics mode implementations
     *
     * Minimal interface - each mode owns its complete timestep behavior.
     * No query methods, no output transformations, just core functionality.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel SPH kernel template (e.g., M4, C2)
     */
    template<class Tvec, template<class> class SPHKernel>
    class PhysicsMode {
        public:
        using Tscal   = shambase::VecComponent<Tvec>;
        using Storage = SolverStorage<Tvec, u32>;
        using Config  = SolverConfig<Tvec, SPHKernel>;

        virtual ~PhysicsMode() = default;

        // ════════════════════════════════════════════════════════════════════════
        // Core - Each mode owns its full timestep
        // ════════════════════════════════════════════════════════════════════════

        /**
         * @brief Execute a complete physics timestep
         *
         * Each mode implements its own sequence:
         * - Newtonian: predictor → tree → omega → gradients → eos → forces → corrector
         * - SR: predictor → tree → omega → gradients → eos → forces → recovery → corrector
         * - MHD: ... → forces → divergence_clean → corrector
         *
         * Mode calls shared operations via callbacks. This avoids combinatorial
         * query methods (requires_X) - mode just does what it needs.
         *
         * @param storage Solver storage
         * @param config Solver configuration
         * @param scheduler Patch scheduler
         * @param dt Timestep size
         * @param callbacks Callbacks for shared operations
         * @return dt_next computed CFL timestep for next iteration
         */
        virtual Tscal evolve_timestep(
            Storage &storage,
            const Config &config,
            PatchScheduler &scheduler,
            Tscal dt,
            const SolverCallbacks<Tscal> &callbacks)
            = 0;

        /**
         * @brief Initialize physics-specific fields in storage
         *
         * Called once during solver setup. Mode allocates any fields it needs.
         * Uses storage field maps for physics-specific fields.
         * Can modify config to set physics-specific flags (use_gradients, etc.)
         */
        virtual void init_fields(Storage &storage, Config &config) = 0;

        // ════════════════════════════════════════════════════════════════════════
        // Metadata
        // ════════════════════════════════════════════════════════════════════════

        virtual std::string_view name() const        = 0;
        virtual std::string_view description() const = 0;

        // ════════════════════════════════════════════════════════════════════════
        // I/O - Declarative field registration
        // ════════════════════════════════════════════════════════════════════════

        /**
         * @brief Get names of fields to export in VTK output
         *
         * Mode stores output-ready values in these fields during computation.
         * VTKDump reads fields by name - no transformation needed.
         *
         * Override to add physics-specific fields (B for MHD, metric for GR).
         */
        virtual std::vector<std::string> get_output_field_names() const {
            return {"density", "pressure", "velocity", "soundspeed"};
        }

        /**
         * @brief Get the physics-specific name for the density field
         *
         * Newtonian: "density" (mass density ρ)
         * SR: "N_labframe" (lab-frame baryon density N)
         */
        virtual const char *get_density_field_name() const = 0;

        /**
         * @brief Get reference to the physics-specific density field
         *
         * Newtonian: storage.density (mass density ρ)
         * SR: storage.density (which aliases N_labframe in SR mode)
         *
         * This allows physics-agnostic modules (ComputeOmega, ComputeGradients)
         * to access the correct density field without knowing the physics.
         */
        virtual shamrock::solvergraph::Field<Tscal> &get_density_field(Storage &storage) const {
            return shambase::get_check_ref(storage.density);
        }

        // ════════════════════════════════════════════════════════════════════════
        // Layout Extension - Physics modes add their fields
        // ════════════════════════════════════════════════════════════════════════

        /**
         * @brief Add physics-specific fields to PatchDataLayerLayout
         *
         * Called after common layout is set. Override to add physics-specific
         * persistent fields (e.g., conserved variables for SR).
         */
        virtual void extend_layout(shamrock::patch::PatchDataLayerLayout &pdl) {}

        /**
         * @brief Add physics-specific fields to ghost layout
         *
         * Called after common ghost layout is set. Override to add physics-specific
         * ghost fields needed for force computation.
         */
        virtual void extend_ghost_layout(shamrock::patch::PatchDataLayerLayout &ghost_layout) {}

        /**
         * @brief Check if mode uses per-particle mass field
         *
         * SR mode with volume-based h uses pmass field. Newtonian uses uniform mass.
         * Used to set config.use_pmass_field after extending layout.
         */
        virtual bool uses_pmass_field() const { return false; }
    };

} // namespace shammodels::gsph::core
