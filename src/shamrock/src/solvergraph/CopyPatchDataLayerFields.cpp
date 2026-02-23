// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CopyPatchDataLayerFields.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the CopyPatchDataLayerFields class for copying fields between patch data layers.
 *
 */

#include "shamrock/solvergraph/CopyPatchDataLayerFields.hpp"

namespace shamrock::solvergraph {

    void CopyPatchDataLayerFields::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        edges.target.set_patchdatas({});

        // Ensures that the layout are all matching sources and targets
        edges.original.get_const_refs().for_each(
            [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                if (pdat.get_layout_ptr().get() != layout_source.get()) {
                    throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
                }
            });

        if (edges.target.get_layout_ptr().get() != layout_target.get()) {
            throw shambase::make_except_with_loc<std::invalid_argument>("layout mismatch");
        }

        // Copy the fields from the original to the target
        edges.target.set_patchdatas(edges.original.get_const_refs().map<patch::PatchDataLayer>(
            [&](u64 id_patch, const patch::PatchDataLayer &pdat) {
                patch::PatchDataLayer pdat_new(layout_target);

                pdat_new.for_each_field_any([&](auto &field) {
                    using T = typename std::remove_reference<decltype(field)>::type::Field_type;
                    field.insert(pdat.get_field<T>(field.get_name()));
                });

                pdat_new.check_field_obj_cnt_match();
                return pdat_new;
            }));
    }

} // namespace shamrock::solvergraph
