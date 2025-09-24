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
 * @brief Extract the ghost layer from the patch data layers.
 */

#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"

namespace shammodels::basegodunov::modules {

    class ExtractGhostLayer : public shamrock::solvergraph::INode {
        std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout;

        public:
        ExtractGhostLayer(std::shared_ptr<shamrock::patch::PatchDataLayerLayout> ghost_layer_layout)
            : ghost_layer_layout(ghost_layer_layout) {}

        struct Edges {
            // inputs
            const shamrock::solvergraph::IPatchDataLayerRefs &patch_data_layers;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
            // outputs
            shamrock::solvergraph::PatchDataLayerDDShared &ghost_layer;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IPatchDataLayerRefs> patch_data_layers,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost,
            std::shared_ptr<shamrock::solvergraph::PatchDataLayerDDShared> ghost_layer) {
            __internal_set_ro_edges({patch_data_layers, idx_in_ghost});
            __internal_set_rw_edges({ghost_layer});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IPatchDataLayerRefs>(0),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(1),
                get_rw_edge<shamrock::solvergraph::PatchDataLayerDDShared>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ExtractGhostLayer"; };

        virtual std::string _impl_get_tex() { return "TODO"; };
    };
} // namespace shammodels::basegodunov::modules
