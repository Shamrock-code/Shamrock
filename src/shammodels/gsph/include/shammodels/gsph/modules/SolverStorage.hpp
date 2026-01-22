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
 * @file SolverStorage.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Storage for GSPH solver runtime data with SolverGraph integration
 *
 * This file contains the storage structure for GSPH solver runtime data,
 * including neighbor caches, ghost data, and field storage.
 *
 * The storage uses the SolverGraph architecture for:
 * - Explicit data dependency tracking
 * - Automatic memory management via free_alloc()
 * - Graph-based operation sequencing
 *
 * The GSPH solver originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"

// GSPH-specific includes
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"

// SolverGraph core includes
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"

// GSPH SolverGraph edge types
#include "shammodels/gsph/solvergraph/GhostCacheEdge.hpp"
#include "shammodels/gsph/solvergraph/GhostHandlerEdge.hpp"
#include "shammodels/gsph/solvergraph/MergedPatchDataEdge.hpp"
#include "shammodels/gsph/solvergraph/SerialPatchTreeEdge.hpp"

// Reuse SPH NeighCache (same implementation)
#include "shammodels/sph/solvergraph/NeighCache.hpp"

// Scheduler and tree includes
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <memory>

namespace shammodels::gsph {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    /**
     * @brief Runtime storage for GSPH solver with SolverGraph integration
     *
     * Stores all temporary data needed during GSPH simulation steps:
     * - SolverGraph for data dependency management
     * - Neighbor caches for particle interactions
     * - Ghost particle data for boundary handling
     * - Computed fields (pressure, sound speed, omega)
     * - Tree structures for neighbor search
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam Tmorton Morton code type for tree construction
     */
    template<class Tvec, class Tmorton>
    struct SolverStorage {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        // Use GSPH ghost handler with Newtonian field names
        using GhostHandle      = gsph::GSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        using RTree = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

        // =====================================================================
        // SolverGraph infrastructure
        // =====================================================================

        /// Central graph for managing edges (data) and nodes (operations)
        shamrock::solvergraph::SolverGraph solver_graph;

        /// Main operation sequence executed each timestep
        std::shared_ptr<shamrock::solvergraph::OperationSequence> solver_sequence;

        // =====================================================================
        // SolverGraph edges - Particle counts and indices
        // =====================================================================

        /// Particle counts per patch (real particles only)
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts;

        /// Particle counts including ghost particles
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts_with_ghost;

        /// Patch rank ownership mapping
        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        // =====================================================================
        // SolverGraph edges - Field references
        // =====================================================================

        /// Position field references with ghost particles
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts;

        /// Smoothing length field references with ghost particles
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts;

        // =====================================================================
        // SolverGraph edges - Neighbor cache (reuse from SPH)
        // =====================================================================

        /// Neighbor cache - uses shamrock's tree-based neighbor search
        std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache;

        // =====================================================================
        // SolverGraph edges - Infrastructure (GSPH-specific)
        // =====================================================================

        /// Serial patch tree for load balancing and interface detection
        std::shared_ptr<solvergraph::SerialPatchTreeEdge<Tvec>> serial_patch_tree;

        /// Ghost handler for boundary particle communication
        std::shared_ptr<solvergraph::GhostHandlerEdge<Tvec>> ghost_handler;

        /// Ghost interface cache for communication optimization
        std::shared_ptr<solvergraph::GhostCacheEdge<Tvec>> ghost_patch_cache;

        /// Merged position-h data for neighbor search (local + ghost)
        std::shared_ptr<solvergraph::MergedPatchDataEdge> merged_xyzh;

        /// Merged patchdata including all ghost fields
        std::shared_ptr<solvergraph::MergedPatchDataEdge> merged_patchdata_ghost;

        // =====================================================================
        // Legacy Component storage (trees - complex lifecycle, migrate later)
        // =====================================================================

        /// Radix trees for neighbor search
        Component<shambase::DistributedData<RTree>> merged_pos_trees;
        Component<shambase::DistributedData<shamtree::KarrasRadixTreeField<Tscal>>>
            rtree_rint_field;

        // =====================================================================
        // SolverGraph edges - Computed fields
        // =====================================================================

        /// Grad-h correction factor (Omega)
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega;

        /// Density field computed via SPH summation
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density;

        /// Pressure from EOS
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure;

        /// Sound speed from EOS
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed;

        /// Gradient fields for MUSCL reconstruction (2nd order)
        /// These are computed when ReconstructConfig::is_muscl() is true
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_density;  ///< \nabla \rho
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_pressure; ///< \nabla P
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vx;       ///< \nabla v_x
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vy;       ///< \nabla v_y
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vz;       ///< \nabla v_z

        /// Old derivatives for predictor-corrector integration
        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

        // =====================================================================
        // Non-graph storage
        // =====================================================================

        /// Ghost data layout pointers
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> xyzh_ghost_layout;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layout;

        /// Minimum h/c_s for CFL timestep calculation
        /// For pure GSPH hydrodynamics: dt_CFL = C_cour * h / c_s
        Tscal h_per_cs_min = std::numeric_limits<Tscal>::max();

        /// Timing statistics
        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;
            f64 riemann   = 0; ///< Time spent in Riemann solver

            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::gsph
