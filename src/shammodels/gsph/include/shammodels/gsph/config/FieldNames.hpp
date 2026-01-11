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
 * @brief Constants for solvergraph edge names in GSPH solver
 *
 * This file defines string constants for solvergraph edge names used in the GSPH solver.
 * Physics field names (xyz, vxyz, hpart, etc.) use standard strings directly,
 * matching the convention used by other SPH methods.
 */

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
     * @brief Physics solvergraph edges
     *
     * These edges reference physics fields. Each physics mode (Newtonian, SR)
     * has its own sub-namespace with appropriate field references.
     */
    namespace physics {

        /**
         * @brief Newtonian physics solvergraph edges
         *
         * These edges reference Newtonian physics fields (xyz, hpart, etc.).
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

        // Future: namespace sr { /* SR physics edges */ }

    } // namespace physics

} // namespace shammodels::gsph::edges
