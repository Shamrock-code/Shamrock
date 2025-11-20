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
 * @file CopyPatchDataLayerFields.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the CopyPatchDataLayerFields class for copying fields between patch data layers.
 *
 */

#include "shambase/exception.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include <memory>

namespace shamrock::solvergraph {

    class CopyPatchDataLayerFields : public INode {

        public:
        CopyPatchDataLayerFields(
            std::shared_ptr<patch::PatchDataLayerLayout> layout_source,
            std::shared_ptr<patch::PatchDataLayerLayout> layout_target)
            : layout_source(layout_source), layout_target(layout_target) {}

        std::shared_ptr<patch::PatchDataLayerLayout> layout_source;
        std::shared_ptr<patch::PatchDataLayerLayout> layout_target;

        struct Edges {
            const IPatchDataLayerRefs &original;
            PatchDataLayerEdge &target;
        };

        void set_edges(
            std::shared_ptr<IPatchDataLayerRefs> original,
            std::shared_ptr<PatchDataLayerEdge> target) {
            __internal_set_ro_edges({original});
            __internal_set_rw_edges({target});
        }

        Edges get_edges() {
            return Edges{get_ro_edge<IPatchDataLayerRefs>(0), get_rw_edge<PatchDataLayerEdge>(0)};
        }

        void _impl_evaluate_internal();

        std::string _impl_get_label() const { return "CopyPatchDataLayerFields"; }

        std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shamrock::solvergraph
