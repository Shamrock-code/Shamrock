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
 * @file GetObjCntFromLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the GetObjCntFromLayer class for extracting object counts from patch data
 * layers.
 *
 */

#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IPatchDataLayerRefs, source)                                       \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::Indexes<u32>, out_ref)

namespace shamrock::solvergraph {

    class GetObjCntFromLayer : public INode {

        public:
        GetObjCntFromLayer() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal() {
            auto edges = get_edges();

            edges.out_ref.indexes = edges.source.get_const_refs().template map<u32>(
                [&](u64 id_patch, const shamrock::patch::PatchDataLayer &pdat) {
                    return pdat.get_obj_cnt();
                });
        }

        std::string _impl_get_label() const { return "GetObjCntFromLayer"; }

        std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shamrock::solvergraph

#undef NODE_EDGES
