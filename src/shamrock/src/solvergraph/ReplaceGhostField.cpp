// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ReplaceGhostField.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief   This module allows replacing ghost values for a generic field with updated values.
 *
 */

#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include "shamrock/patch/PatchDataField.hpp"

namespace shamrock::solvergraph {
    template<class T>
    void shamrock::solvergraph::ReplaceGhostField<T>::_impl_evaluate_internal() {

        StackEntry stack_loc{};
        auto edges = get_edges();

        auto &ghost_fields = edges.ghost_fields;
        auto &fields       = edges.fields;

        std::map<u32, u32> gz_map;
        ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, const PatchDataField<T> &pdat_field) {
            gz_map[r] += pdat_field.get_obj_cnt();
        });

        // remove old fields
        fields.get_refs().for_each([&](u32 id_patch, PatchDataField<T> &field) {
            // TODO: currently we guess the GZ size using the input, we should use in place ghost zones really ...
            field.shrink(gz_map.at(id_patch));
        });

        // replace new fields
        ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, const PatchDataField<T> &pdat_field) {
            fields.get_field(r).insert(pdat_field);
        });
    }

    template class shamrock::solvergraph::ReplaceGhostField<f64>;
    template class shamrock::solvergraph::ReplaceGhostField<f64_3>;

} // namespace shamrock::solvergraph
