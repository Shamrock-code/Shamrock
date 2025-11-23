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
 * @file CopyPatchDataFieldFromLayer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the CopyPatchDataFieldFromLayer class for copying fields between patch data layers.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/exception.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/IPatchDataLayerRefs.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include <memory>

namespace shamrock::solvergraph {

    template<class T>
    class CopyPatchDataFieldFromLayer : public INode {

        u32 field_idx;

        public:
        CopyPatchDataFieldFromLayer(u32 field_idx) : field_idx(field_idx) {}

        CopyPatchDataFieldFromLayer(
            shamrock::patch::PatchDataLayerLayout &layout, const std::string &field_name)
            : CopyPatchDataFieldFromLayer(layout.get_field_idx<T>(field_name)) {}

        CopyPatchDataFieldFromLayer(
            const std::shared_ptr<shamrock::patch::PatchDataLayerLayout> &layout,
            const std::string &field_name)
            : CopyPatchDataFieldFromLayer(shambase::get_check_ref(layout), field_name) {}

        struct Edges {
            const IPatchDataLayerRefs &original;
            IFieldRefs<T> &target;
        };

        void set_edges(
            std::shared_ptr<IPatchDataLayerRefs> original,
            std::shared_ptr<IFieldRefs<T>> target) {
            __internal_set_ro_edges({original});
            __internal_set_rw_edges({target});
        }

        Edges get_edges() {
            return Edges{get_ro_edge<IPatchDataLayerRefs>(0), get_rw_edge<IFieldRefs<T>>(0)};
        }

        inline void _impl_evaluate_internal(){

            auto edges = get_edges();

            shambase::DistributedData<u32> sizes {};

            edges.original.get_const_refs().for_each(
                [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                    sizes.add_obj(id_patch, pdat.get_obj_cnt());
                });

            edges.target.ensure_sizes(sizes);

            edges.original.get_const_refs().for_each(
                [&](u64 id_patch, const patch::PatchDataLayer &source) {
                    PatchDataField<T> & dest = edges.target.get_refs().get(id_patch).get();
                    dest.overwrite(source.get_field<T>(field_idx), source.get_obj_cnt());
                });
        }

        std::string _impl_get_label() { return "CopyPatchDataFieldFromLayer"; }

        std::string _impl_get_tex() { return "TODO"; }
    };

    template class CopyPatchDataFieldFromLayer<float>;
} // namespace shamrock::solvergraph
