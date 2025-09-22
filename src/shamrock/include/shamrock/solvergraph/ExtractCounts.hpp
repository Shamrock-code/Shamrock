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
 * @file ExtractCounts.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the ExtractCounts class for extracting object counts from patch data layer
 * references.
 *
 */

#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shamrock::solvergraph {

    class ExtractCounts : public INode {

        public:
        ExtractCounts() {}

        struct Edges {
            const IPatchDataLayerRefs &refs;
            Indexes<u32> &counts;
        };

        void set_edges(
            std::shared_ptr<IPatchDataLayerRefs> refs, std::shared_ptr<Indexes<u32>> counts) {
            __internal_set_ro_edges({refs});
            __internal_set_rw_edges({counts});
        }

        Edges get_edges() {
            return Edges{get_ro_edge<IPatchDataLayerRefs>(0), get_rw_edge<Indexes<u32>>(0)};
        }

        void _impl_evaluate_internal() {
            auto edges           = get_edges();
            edges.counts.indexes = edges.refs.get_const_refs().map<u32>(
                [](u64 id_patch, const patch::PatchDataLayer &pdat) {
                    return pdat.get_obj_cnt();
                });
        }

        std::string _impl_get_label() { return "ExtractCounts"; }

        std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shamrock::solvergraph
