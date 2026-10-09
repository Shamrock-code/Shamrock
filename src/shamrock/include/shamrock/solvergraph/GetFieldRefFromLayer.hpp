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
 * @file GetFieldRefFromLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the GetFieldRefFromLayer class for extracting field references from patch data
 * layers.
 *
 */

#include "shambase/memory.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IPatchDataLayerRefs, source)                                       \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::FieldRefs<T>, out_ref)

namespace shamrock::solvergraph {

    template<class T>
    class GetFieldRefFromLayer : public INode {

        u32 field_idx;

        public:
        GetFieldRefFromLayer(u32 field_idx) : field_idx(field_idx) {}

        GetFieldRefFromLayer(
            shamrock::patch::PatchDataLayerLayout &layout, const std::string &field_name)
            : GetFieldRefFromLayer(layout.get_field_idx<T>(field_name)) {}

        GetFieldRefFromLayer(
            const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &layout,
            const std::string &field_name)
            : GetFieldRefFromLayer(shambase::get_check_ref(layout), field_name) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal() {
            auto edges = get_edges();

            edges.out_ref.set_refs(
                edges.source.get_const_refs()
                    .template map<std::reference_wrapper<PatchDataField<T>>>(
                        [&](u64 id_patch, shamrock::patch::PatchDataLayer &pdat) {
                            return std::ref(pdat.get_field<T>(field_idx));
                        }));
        }

        std::string _impl_get_label() const { return "GetFieldRefFromLayer"; }

        std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shamrock::solvergraph

#undef NODE_EDGES
