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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRCellInfos.hpp"
#include "shammodels/common/amr/AMRStencilCache.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/common/amr/NeighGraphLinkField.hpp"
#include "shammodels/ramses/GhostZoneData.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/NodeBuildTrees.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shammodels/ramses/solvegraph/TreeEdge.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"
#include <memory>

namespace shammodels::basegodunov {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    template<class Tvec, class TgridVec, class Tmorton_>
    class SolverStorage {
        public:
        using Tmorton   = Tmorton_;
        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using RTree = RadixTree<Tmorton, TgridVec>;

        shamrock::solvergraph::SolverGraph solver_graph;

        std::shared_ptr<shamrock::solvergraph::IDataEdge<std::vector<u64>>> local_patch_ids;

        std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> patch_rank_owner;

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt_over2;

        std::shared_ptr<shamrock::solvergraph::FieldRefs<TgridVec>> refs_block_min;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<TgridVec>> refs_block_max;

        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> block_counts;
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> block_counts_with_ghost;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> refs_rho;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> refs_rhov;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> refs_rhoe;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> refs_rho_dust;
        std::shared_ptr<shamrock::solvergraph::FieldRefs<Tvec>> refs_rhov_dust;

        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> vel;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> press;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> vel_dust;

        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> block_cell_sizes;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> cell0block_aabb_lower;
        std::shared_ptr<shamrock::solvergraph::Field<TgridUint>> block_levels;

        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_rho;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dx_v;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dy_v;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dz_v;
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_P;
        /// dust fields gradients (grad rho_dust)
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> grad_rho_dust;
        /// dust fields gradients (d vdust / d x)
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dx_v_dust;
        /// dust fields gradients (d vdust / d y)
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dy_v_dust;
        /// dust fields gradients (d vdust / d z)
        std::shared_ptr<shamrock::solvergraph::Field<Tvec>> dz_v_dust;

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> rho_mean;
        std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> simulation_volume;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> cell_mass;

        Component<shambase::DistributedData<shammath::AABB<TgridVec>>> merge_patch_bounds;

        std::shared_ptr<solvergraph::TreeEdge<Tmorton, TgridVec>> trees;

        std::shared_ptr<shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>
            block_graph_edge;

        std::shared_ptr<shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>
            cell_graph_edge;

        std::shared_ptr<shamrock::solvergraph::OperationSequence> solver_sequence;

        std::shared_ptr<shamrock::solvergraph::OperationSequence> node_cons_to_prim;
        std::shared_ptr<modules::NodeBuildTrees<Tmorton, TgridVec>> build_trees;

        Component<SerialPatchTree<TgridVec>> serial_patch_tree;

        Component<GhostZonesData<Tvec, TgridVec>> ghost_zone_infos;

        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layout;

        std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> source_patches;

        std::shared_ptr<shamrock::solvergraph::PatchDataLayerEdge> merged_patchdata_ghost;

        std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> exchange_gz_edge;

        std::shared_ptr<shamrock::solvergraph::DDSharedScalar<modules::GhostLayerCandidateInfos>>
            ghost_layers_candidates_edge;

        std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box_edge;

        std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> rho_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>> press_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_dust_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_dust_face_zm;

        Component<shamrock::ComputeField<Tscal>> dtrho;
        Component<shamrock::ComputeField<Tvec>> dtrhov;
        Component<shamrock::ComputeField<Tscal>> dtrhoe;

        Component<shamrock::ComputeField<Tscal>> rho_next_no_drag;
        Component<shamrock::ComputeField<Tvec>> rhov_next_no_drag;
        Component<shamrock::ComputeField<Tscal>> rhoe_next_no_drag;

        // next time step dust density before drag
        Component<shamrock::ComputeField<Tscal>> rho_d_next_no_drag;
        // next time step dust momentum before drag
        Component<shamrock::ComputeField<Tvec>> rhov_d_next_no_drag;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tscal, 2>>>
            rho_dust_face_zm;

        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_xp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_xm;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_yp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_ym;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_zp;
        std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<std::array<Tvec, 2>>> vel_dust_face_zm;

        // time derivative dust density
        Component<shamrock::ComputeField<Tscal>> dtrho_dust;
        // time derivative dust momemtum
        Component<shamrock::ComputeField<Tvec>> dtrhov_dust;

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;

            /// Reset the timings logged in the storage
            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::basegodunov
