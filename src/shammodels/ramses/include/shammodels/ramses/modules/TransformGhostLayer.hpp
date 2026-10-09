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
 * @file TransformGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<shammath::AABB<TgridVec>>, sim_box)                     \
    X_RO(shamrock::solvergraph::DDSharedScalar<GhostLayerCandidateInfos>, ghost_layers_candidates) \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::PatchDataLayerDDShared, ghost_layer)

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

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "TransformGhostLayer"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
