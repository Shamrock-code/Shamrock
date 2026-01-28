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
#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
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

        // Use GSPH ghost handler with Newtonian field names
        using GhostHandle      = gsph::GSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        using RTree = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

        std::shared_ptr<shamrock::solvergraph::ExchangeGhostLayer> exchange_gz_node;
        std::shared_ptr<shamrock::solvergraph::ExchangeGhostLayer> exchange_gz_positions;

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

        /// Serial patch tree for load balancing
        Component<SerialPatchTree<Tvec>> serial_patch_tree;

        /// Ghost handler for boundary particles
        Component<GhostHandle> ghost_handler;
        Component<GhostHandleCache> ghost_patch_cache;

        /// Merged position-h data for neighbor search
        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>> merged_xyzh;

        /// Radix trees for neighbor search
        Component<shambase::DistributedData<RTree>> merged_pos_trees;
        Component<shambase::DistributedData<shamtree::KarrasRadixTreeField<Tscal>>>
            rtree_rint_field;

        /// Grad-h correction factor (Omega)
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega;

        /// Ghost data layout and merged data
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> xyzh_ghost_layout;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layout;
        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>>
            merged_patchdata_ghost;

        /// Density field computed via SPH summation
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> density;

        /// Thermodynamic fields computed from EOS
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed;

        /// Gradient fields for MUSCL reconstruction (2nd order)
        /// These are computed when ReconstructConfig::is_muscl() is true
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_density;  ///< \nabla \rho
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_pressure; ///< \nabla P
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vx;       ///< \nabla v_x
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vy;       ///< \nabla v_y
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_vz;       ///< \nabla v_z

        /// Minimum h/c_s for CFL timestep calculation
        /// For pure GSPH hydrodynamics: dt_CFL = C_cour * h / c_s
        Tscal h_per_cs_min = std::numeric_limits<Tscal>::max();

        /// Old derivatives for predictor-corrector integration
        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

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
