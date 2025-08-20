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
 * @file FuseGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief A solver graph node to fuse a ghost layer into a set of patch data layers.
 */

#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"

namespace shammodels::basegodunov::modules {

    class FuseGhostLayer : public shamrock::solvergraph::INode {

        public:
        FuseGhostLayer() {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
            // outputs
            shamrock::solvergraph::IPatchDataLayerRefs &patch_data_layers;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer,
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> patch_data_layers) {
            __internal_set_ro_edges({ghost_layer});
            __internal_set_rw_edges({patch_data_layers});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
                get_rw_edge<shamrock::solvergraph::IPatchDataLayerRefs>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "FuseGhostLayer"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
