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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/CompressedLeafBVH.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <memory>

namespace shammodels::sph {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    template<class Tvec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using GhostHandle      = BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache = typename GhostHandle::CacheMap;

        using RTree = shamtree::CompressedLeafBVH<Tmorton, Tvec, 3>;

        shamrock::solvergraph::SolverGraph solver_graph;
        std::shared_ptr<shamrock::solvergraph::OperationSequence> solver_sequence;

        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts;
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts_with_ghost;

        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> positions_with_ghosts;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> hpart_with_ghosts;

        std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache;

        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        Component<SerialPatchTree<Tvec>> serial_patch_tree;

        Component<GhostHandle> ghost_handler;

        Component<GhostHandleCache> ghost_patch_cache;

        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>> merged_xyzh;

        Component<shambase::DistributedData<RTree>> merged_pos_trees;

        Component<shambase::DistributedData<shamtree::KarrasRadixTreeField<Tscal>>>
            rtree_rint_field;

        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> omega;

        // xi resemble omega for the managment in the code, if somethign is done to omega and not to
        // xi (when enabled) it is a bug !
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> xi;

        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layout;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> xyzh_ghost_layout;

        Component<shambase::DistributedData<shamrock::patch::PatchDataLayer>>
            merged_patchdata_ghost;

        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> alpha_av_updated;
        Component<shambase::DistributedData<PatchDataField<Tscal>>> alpha_av_ghost;

        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> pressure;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> soundspeed;

        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

        Component<shamrock::ComputeField<Tvec>> old_dB_on_rho;
        Component<shamrock::ComputeField<Tscal>> old_dpsi_on_ch;

        Component<shamrock::ComputeField<Tscal>> old_dtepsilon;
        Component<shamrock::ComputeField<Tvec>> old_dtdeltav;

        Component<std::vector<SinkParticle<Tvec>>> sinks;

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;

            /// Reset the timings logged in the storage
            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::sph
