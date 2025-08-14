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
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
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
            const PatchDataLayerRefs &original;
            PatchDataLayerEdge &target;
        };

        void set_edges(
            std::shared_ptr<PatchDataLayerRefs> original,
            std::shared_ptr<PatchDataLayerEdge> target) {
            __internal_set_ro_edges({original});
            __internal_set_rw_edges({target});
        }

        Edges get_edges() {
            return Edges{get_ro_edge<PatchDataLayerRefs>(0), get_rw_edge<PatchDataLayerEdge>(0)};
        }

        void _impl_evaluate_internal() {
            auto edges = get_edges();

            // Ensures that the layout are all matching sources and targets
            edges.original.patchdatas.for_each([&](u64 id_patch, patch::PatchDataLayer &pdat) {
                if (pdat.get_layout_ptr().get() != layout_source.get()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
                }
            });

            if (edges.target.layout.get() != layout_target.get()) {
                throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
            }

            // Copy the fields from the original to the target
            edges.target.patchdatas = edges.original.patchdatas.map<patch::PatchDataLayer>(
                [&](u64 id_patch, patch::PatchDataLayer &pdat) {
                    patch::PatchDataLayer pdat_new(layout_target);

                    pdat_new.for_each_field_any([&](auto &field) {
                        using T = typename std::remove_reference<decltype(field)>::type::Field_type;
                        field.insert(pdat.get_field<T>(field.get_name()));
                    });

                    pdat_new.check_field_obj_cnt_match();
                    return pdat_new;
                });
        }

        std::string _impl_get_label() { return "CopyPatchDataLayerFields"; }

        std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shamrock::solvergraph
