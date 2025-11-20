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
 * @file TransformGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class TransformGhostLayer : public shamrock::solvergraph::INode {

        GhostLayerGenMode mode;
        bool transform_vec_x = true;
        bool transform_vec_y = true;
        bool transform_vec_z = true;
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        TransformGhostLayer(
            GhostLayerGenMode mode,
            bool transform_vec_x,
            bool transform_vec_y,
            bool transform_vec_z,
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : mode(mode), transform_vec_x(transform_vec_x), transform_vec_y(transform_vec_y),
              transform_vec_z(transform_vec_z), ghost_layer_layout(ghost_layer_layout) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>> &sim_box;
            const shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>
                &ghost_layers_candidates;
            // outputs
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>> sim_box,
            std::shared_ptr<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>
                ghost_layers_candidates,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer) {
            __internal_set_ro_edges({sim_box, ghost_layers_candidates});
            __internal_set_rw_edges({ghost_layer});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>>(0),
                get_ro_edge<shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>>(1),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "TransformGhostLayer"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules
