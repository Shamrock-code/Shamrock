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
 * @file SolverStorage.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Storage for GSPH solver runtime data
 *
 * This file contains the storage structure for GSPH solver runtime data,
 * including neighbor caches, ghost data, and field storage.
 *
 * The GSPH solver originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <unordered_map>
#include <memory>
#include <string>

namespace shammodels::gsph {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    /**
     * @brief Runtime storage for GSPH solver
     *
     * Stores all temporary data needed during GSPH simulation steps:
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

        // Reuse SPH ghost handler - the mechanism is the same
        using GhostHandle      = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        using RTree = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

        //////////////////////////////////////////////////////////////////////
        // Solver Graph (SSOT registry for edges and nodes)
        //////////////////////////////////////////////////////////////////////

        /// Central registry for all solver edges and nodes
        shamrock::solvergraph::SolverGraph solver_graph;

        //////////////////////////////////////////////////////////////////////
        // Field Maps (physics-agnostic field registry)
        //////////////////////////////////////////////////////////////////////

        /// Scalar fields registered by name (physics modes register their fields here)
        std::unordered_map<std::string, std::shared_ptr<shamrock::solvergraph::Field<Tscal>>>
            scalar_fields;

        /// Vector fields registered by name
        std::unordered_map<std::string, std::shared_ptr<shamrock::solvergraph::Field<Tvec>>>
            vector_fields;

        //////////////////////////////////////////////////////////////////////
        // Particle Management (physics-agnostic)
        //////////////////////////////////////////////////////////////////////

        /// Particle counts per patch
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts;
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts_with_ghost;

        /// Position and smoothing length fields with ghosts
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts;

        /// Neighbor cache - uses shamrock's tree-based neighbor search
        std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache;

        /// Patch rank ownership
        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        //////////////////////////////////////////////////////////////////////
        // Tree Structures (physics-agnostic)
        //////////////////////////////////////////////////////////////////////

        /// Serial patch tree for load balancing
        Component<SerialPatchTree<Tvec>> serial_patch_tree;

        /// Merged position-h data for neighbor search
        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>> merged_xyzh;

        /// Radix trees for neighbor search
        Component<shambase::DistributedData<RTree>> merged_pos_trees;
        Component<shambase::DistributedData<shamtree::KarrasRadixTreeField<Tscal>>>
            rtree_rint_field;

        //////////////////////////////////////////////////////////////////////
        // Ghost Handling (physics-agnostic)
        //////////////////////////////////////////////////////////////////////

        /// Ghost handler for boundary particles
        Component<GhostHandle> ghost_handler;
        Component<GhostHandleCache> ghost_patch_cache;

        /// Ghost data layout and merged data
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> xyzh_ghost_layout;
        Component<std::shared_ptr<shamrock::patch::PatchDataLayerLayout>> ghost_layout;
        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>>
            merged_patchdata_ghost;

        //////////////////////////////////////////////////////////////////////
        // SPH Kernel Fields (physics-agnostic)
        //////////////////////////////////////////////////////////////////////

        /// Grad-h correction factor (Omega) - always computed via SPH summation
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega;

        //////////////////////////////////////////////////////////////////////
        // Density Field (physics-specific, set by PhysicsMode::init_fields)
        // Newtonian: density (mass density ρ = m × Σ W)
        // SR: N_labframe (lab-frame baryon density N = ν × Σ W)
        //////////////////////////////////////////////////////////////////////
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density;

        //////////////////////////////////////////////////////////////////////
        // Thermodynamic Fields (computed from EOS)
        //////////////////////////////////////////////////////////////////////

        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed;

        //////////////////////////////////////////////////////////////////////
        // Gradient Fields for MUSCL Reconstruction
        //////////////////////////////////////////////////////////////////////

        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_density;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_pressure;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vx;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vy;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vz;

        //////////////////////////////////////////////////////////////////////
        // CFL Timestep
        //////////////////////////////////////////////////////////////////////

        /// Minimum h/c_s for CFL timestep calculation
        Tscal h_per_cs_min = std::numeric_limits<Tscal>::max();

        //////////////////////////////////////////////////////////////////////
        // SR-GSPH Specific Fields
        // These are only used when physics_mode is SR
        //////////////////////////////////////////////////////////////////////

        /// Lorentz factor γ = 1/√(1-v²)
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> gamma_lorentz;

        /// Relativistic enthalpy H = 1 + u + P/n
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> enthalpy;

        /// Lab-frame volume V = ν/N
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> V_labframe;

        /// Baryon number per particle ν
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> nu_baryon;

        /// Conserved momentum S = γHv
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> S_momentum;

        /// Conserved energy e = γH - P/N
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> e_energy;

        /// Momentum derivative dS/dt
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dS_momentum;

        /// Energy derivative de/dt
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> de_energy;

        //////////////////////////////////////////////////////////////////////
        // Integration State
        //////////////////////////////////////////////////////////////////////

        /// Flag: has physics mode initialized its fields?
        bool physics_fields_initialized = false;

        /// Flag: has SR been initialized (prim2cons done)?
        bool sr_initialized = false;

        /// Old derivatives for predictor-corrector (SR)
        Component<shamrock::ComputeField<Tvec>> old_dS;
        Component<shamrock::ComputeField<Tscal>> old_de;

        /// Old derivatives for predictor-corrector (Newtonian)
        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

        //////////////////////////////////////////////////////////////////////
        // Timing Statistics
        //////////////////////////////////////////////////////////////////////

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;
            f64 riemann   = 0;

            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::gsph
