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
 * @file ExchangeGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

namespace shammodels::basegodunov::modules {

    class ExchangeGhostLayer : public shamrock::solvergraph::INode {
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        ExchangeGhostLayer(
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : ghost_layer_layout(ghost_layer_layout) {}

        struct Edges {
            const shamrock::solvergraph::ScalarsEdge<u32> &rank_owner;
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> rank_owner,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer) {
            __internal_set_ro_edges({rank_owner});
            __internal_set_rw_edges({ghost_layer});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<u32>>(0),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ExchangeGhostLayer"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
