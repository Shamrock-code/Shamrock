// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CopyPatchDataField.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the CopyPatchDataField class for copying fields between patch data field
 * references.
 *
 */

#include "shamrock/solvergraph/CopyPatchDataField.hpp"

namespace shamrock::solvergraph {

    template<class T>
    void CopyPatchDataField<T>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();
        // logger::raw_ln("Cpy_Dat: \t", &edges.original, "-", &edges.target,"\n");

        // Collect size information from source fields
        shambase::DistributedData<u32> sizes = {};

        edges.original.get_refs().for_each([&](u64 id_patch, const PatchDataField<T> &field) {
            sizes.add_obj(id_patch, u32(field.get_obj_cnt()));
        });

        // Ensure target field has correct size for each patch
        edges.target.ensure_sizes(sizes);

        // Copy the fields from the original to the target
        edges.target.get_refs().for_each([&](u64 id_patch, PatchDataField<T> &field) {
            auto &source_field = edges.original.get_field(id_patch);
            if (field.get_nvar() != source_field.get_nvar()) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "nvar mismatch between source and target fields for patch "
                    + std::to_string(id_patch)
                    + ". Source nvar: " + std::to_string(source_field.get_nvar())
                    + ", Target nvar: " + std::to_string(field.get_nvar()));
            }
            field.overwrite(source_field, source_field.get_obj_cnt());
        });
    }

    template<class T>
    std::string CopyPatchDataField<T>::_impl_get_tex() {
        std::string tmp = "Copy field ${original} to ${target}";
        shambase::replace_all(tmp, "{original}", get_ro_edge_base(0).get_tex_symbol());
        shambase::replace_all(tmp, "{target}", get_rw_edge_base(0).get_tex_symbol());
        return tmp;
    }

    template class CopyPatchDataField<f64>;
    template class CopyPatchDataField<f64_3>;

} // namespace shamrock::solvergraph
