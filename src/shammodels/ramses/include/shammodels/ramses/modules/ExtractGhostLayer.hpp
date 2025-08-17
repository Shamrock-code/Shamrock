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
 * @file ExtractGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class ExtractGhostLayer : public shamrock::solvergraph::INode {

        public:
        ExtractGhostLayer(
            GhostLayerGenMode mode,
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : mode(mode), ghost_layer_layout(ghost_layer_layout) {}

        private:
        GhostLayerGenMode mode;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        struct Edges {
            // inputs
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>> &sim_box;
            const shamrock::solvergraph::PatchDataLayerRefs &patch_data_layers;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
            // outputs
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerRefs> patch_data_layers,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer) {
            __internal_set_ro_edges({sim_box, patch_data_layers, idx_in_ghost});
            __internal_set_rw_edges({ghost_layer});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::PatchDataLayerRefs>(1),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(2),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ExtractGhostLayer"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
