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
 * @file FindGhostLayerIndices.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    class FindGhostLayerIndices : public shamrock::solvergraph::INode {

        GhostLayerGenMode mode;

        public:
        FindGhostLayerIndices(GhostLayerGenMode mode) : mode(mode) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>> &sim_box;
            const shamrock::solvergraph::IPatchDataLayerRefs &patch_data_layers;
            const shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>
                &ghost_layers_candidates;
            const shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>> &patch_boxes;
            // outputs
            shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> patch_data_layers,
            std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
                ghost_layers_candidates,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>
                patch_boxes,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost) {
            __internal_set_ro_edges(
                {sim_box, patch_data_layers, ghost_layers_candidates, patch_boxes});
            __internal_set_rw_edges({idx_in_ghost});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::IPatchDataLayerRefs>(1),
                get_ro_edge<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(2),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<shammath::AABB<TgridVec>>>(3),
                get_rw_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "FindGhostLayerIndices"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules
