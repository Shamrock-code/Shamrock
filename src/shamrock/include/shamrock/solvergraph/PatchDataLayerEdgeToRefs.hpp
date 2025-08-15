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
 * @file PatchDataLayerEdgeToRefs.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the PatchDataLayerEdgeToRefs class for converting patch data layer edges to
 * references.
 *
 */

#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"

namespace shamrock::solvergraph {

    class PatchDataLayerEdgeToRefs : public INode {

        public:
        PatchDataLayerEdgeToRefs() {}

        /// Utility struct to list the refs to the edges
        struct Edges {
            PatchDataLayerEdge &original;
            PatchDataLayerRefs &refs;
        };

        /**
         * @brief Set the edges of the node
         *
         * Set the edge that will be freed by this node
         *
         * @param to_free The node to free
         */
        inline void set_edges(
            std::shared_ptr<PatchDataLayerEdge> original,
            std::shared_ptr<PatchDataLayerRefs> refs) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({refs, original});
        }

        /// Get the edges of the node
        inline Edges get_edges() {
            return Edges{get_rw_edge<PatchDataLayerEdge>(0), get_rw_edge<PatchDataLayerRefs>(1)};
        }

        /// Evaluate the node
        inline void _impl_evaluate_internal() {

            auto edges = get_edges();
            edges.refs.patchdatas
                = edges.original.patchdatas.map<std::reference_wrapper<patch::PatchDataLayer>>(
                    [](u64 id_patch, patch::PatchDataLayer &pdat) {
                        return std::ref(pdat);
                    });
        }

        /// Get the label of the node
        inline virtual std::string _impl_get_label() { return "PatchDataLayerEdgeToRefs"; };

        /// Get the TeX representation of the node
        inline virtual std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shamrock::solvergraph
