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
 * @file FuseGhostLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief A solver graph node to fuse a ghost layer into a set of patch data layers.
 */

#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/PatchDataLayerDDShared.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::PatchDataLayerDDShared, ghost_layer)                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IPatchDataLayerRefs, patch_data_layers)

namespace shammodels::basegodunov::modules {

    class FuseGhostLayer : public shamrock::solvergraph::INode {

        public:
        FuseGhostLayer() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "FuseGhostLayer"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
