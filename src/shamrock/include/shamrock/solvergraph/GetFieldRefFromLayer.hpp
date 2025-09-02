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

#include "shambase/memory.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include <memory>

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

        struct Edges {
            const IPatchDataLayerRefs &source;
            shamrock::solvergraph::FieldRefs<T> &out_ref;
        };

        void set_edges(
            std::shared_ptr<IPatchDataLayerRefs> source,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<T>> out_ref) {
            __internal_set_ro_edges({source});
            __internal_set_rw_edges({out_ref});
        }

        Edges get_edges() {
            return Edges{
                get_ro_edge<IPatchDataLayerRefs>(0),
                get_rw_edge<shamrock::solvergraph::FieldRefs<T>>(0)};
        }

        void _impl_evaluate_internal() {
            auto edges = get_edges();

            edges.out_ref.set_refs(
                edges.source.get_const_refs()
                    .template map<std::reference_wrapper<PatchDataField<T>>>(
                        [&](u64 id_patch, shamrock::patch::PatchDataLayer &pdat) {
                            return std::ref(pdat.get_field<T>(field_idx));
                        }));
        }

        std::string _impl_get_label() { return "GetFieldRefFromLayer"; }

        std::string _impl_get_tex() { return "TODO"; }
    };
} // namespace shamrock::solvergraph
