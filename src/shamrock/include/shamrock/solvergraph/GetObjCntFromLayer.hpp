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
 * @file GetFieldRefFromLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the GetFieldRefFromLayer class for extracting field references from patch data
 * layers.
 *
 */

#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

namespace shamrock::solvergraph {

    template<class T>
    class GetObjCntFromLayer : public INode {

        u32 field_idx;

        public:
        GetObjCntFromLayer() {}

        struct Edges {
            const IPatchDataLayerRefs &source;
            shamrock::solvergraph::Indexes<u32> &out_ref;
        };

        void set_edges(
            std::shared_ptr<IPatchDataLayerRefs> source,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> out_ref) {
            __internal_set_ro_edges({source});
            __internal_set_rw_edges({out_ref});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<IPatchDataLayerRefs>(0),
                get_rw_edge<shamrock::solvergraph::Indexes<u32>>(0)};
        }

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
