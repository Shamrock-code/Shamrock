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
 * @file FieldNames.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Constants for field names in GSPH solver, organized by physics mode
 *
 * This file defines PatchDataField names used in PatchDataLayerLayout.
 * Fields are organized by physics mode to clearly separate:
 * - Common fields (used by all physics modes)
 * - Newtonian-specific fields
 * - SR (Special Relativity) specific fields
 *
 * Solvergraph edge names are also organized by physics mode in the edges:: namespace.
 */

namespace shammodels::gsph::names {

    // ========================================================================
    // Common fields - used by ALL physics modes
    // ========================================================================
    namespace common {

        /// Position field (3D coordinates)
        inline constexpr const char *xyz = "xyz";

        /// Smoothing length field
        inline constexpr const char *hpart = "hpart";

    } // namespace common

    // ========================================================================
    // Newtonian physics fields
    // ========================================================================
    namespace newtonian {

        /// 3-velocity field
        inline constexpr const char *vxyz = "vxyz";

        /// 3-acceleration field
        inline constexpr const char *axyz = "axyz";

        /// Specific internal energy u
        inline constexpr const char *uint = "uint";

        /// Time derivative of internal energy du/dt
        inline constexpr const char *duint = "duint";

        /// Density ρ (derived from h)
        inline constexpr const char *density = "density";

        /// Pressure P (derived from EOS)
        inline constexpr const char *pressure = "pressure";

        /// Sound speed c_s (derived from EOS)
        inline constexpr const char *soundspeed = "soundspeed";

        /// Grad-h correction factor Ω
        inline constexpr const char *omega = "omega";

        /// Gradient of density ∇ρ (for MUSCL reconstruction)
        inline constexpr const char *grad_density = "grad_density";

        /// Gradient of pressure ∇P (for MUSCL reconstruction)
        inline constexpr const char *grad_pressure = "grad_pressure";

        /// Gradient of velocity x-component ∇v_x (for MUSCL reconstruction)
        inline constexpr const char *grad_vx = "grad_vx";

        /// Gradient of velocity y-component ∇v_y (for MUSCL reconstruction)
        inline constexpr const char *grad_vy = "grad_vy";

        /// Gradient of velocity z-component ∇v_z (for MUSCL reconstruction)
        inline constexpr const char *grad_vz = "grad_vz";

    } // namespace newtonian

    // ========================================================================
    // Special Relativity physics fields (for future SR implementation)
    // ========================================================================
    namespace sr {

        /// 4-velocity spatial components u^i
        inline constexpr const char *ux = "ux";

        /// Lorentz factor γ = u^0
        inline constexpr const char *lorentz = "lorentz";

        /// Rest frame density ρ_0
        inline constexpr const char *rho_rest = "rho_rest";

        /// Lab frame density ρ = γρ_0
        inline constexpr const char *rho_lab = "rho_lab";

        /// Specific enthalpy h = 1 + ε + P/ρ
        inline constexpr const char *enthalpy = "enthalpy";

        /// Specific internal energy ε
        inline constexpr const char *eps = "eps";

        /// Time derivative of specific internal energy dε/dt
        inline constexpr const char *deps = "deps";

    } // namespace sr

} // namespace shammodels::gsph::names

// ============================================================================
// Solvergraph edge names
// ============================================================================
namespace shammodels::gsph::edges {

    /**
     * @brief Infrastructure solvergraph edges
     *
     * These edges handle computational infrastructure: particle counting,
     * MPI distribution, neighbor finding, etc. They are physics-independent.
     */
    namespace infra {

        /// Particle counts per patch
        inline constexpr const char *part_counts = "part_counts";

        /// Particle counts including ghosts
        inline constexpr const char *part_counts_with_ghost = "part_counts_with_ghost";

        /// Patch rank ownership
        inline constexpr const char *patch_rank_owner = "patch_rank_owner";

        /// Neighbor cache
        inline constexpr const char *neigh_cache = "neigh_cache";

        /// Temporary sizes for h-iteration
        inline constexpr const char *sizes = "sizes";

    } // namespace infra

    /**
     * @brief Newtonian physics solvergraph edges
     */
    namespace newtonian {

        /// Position references with ghosts
        inline constexpr const char *positions_with_ghosts = "part_pos";

        /// Smoothing length references with ghosts
        inline constexpr const char *hpart_with_ghosts = "h_part";

        /// Position merged references (for h-iteration)
        inline constexpr const char *pos_merged = "pos";

        /// Old smoothing length references (for h-iteration)
        inline constexpr const char *h_old = "h_old";

        /// New smoothing length references (for h-iteration)
        inline constexpr const char *h_new = "h_new";

        /// Epsilon h references (for h-iteration convergence)
        inline constexpr const char *eps_h = "eps_h";

    } // namespace newtonian

    /**
     * @brief Special Relativity physics solvergraph edges (for future SR implementation)
     */
    namespace sr {

        // Future SR-specific edges will be added here

    } // namespace sr

} // namespace shammodels::gsph::edges
