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
 * @file ReplaceGhostFields.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shamrock/solvergraph/ReplaceGhostFields.hpp"
#include "shamrock/patch/PatchDataField.hpp"

template<class T>
void shamrock::solvergraph::ReplaceGhostFields<T>::_impl_evaluate_internal() {

    StackEntry stack_loc{};
    auto edges = get_edges();

    auto &ghost_fields = edges.ghost_fields;
    auto &fields       = edges.fields;

    std::map<u32, u32> gz_map;
    ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
        gz_map[r] += pdat_field.get_obj_cnt();
    });

    fields.get_refs().for_each([&](u32 id_patch, PatchDataField<T> &field) {
        PatchDataField<T> temp_pdat(field.get_name(), fields.get_nvar());
        ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
            if (r == id_patch) {
                temp_pdat.insert(pdat_field);
            }
        });
        field.resize(field.get_obj_cnt() - gz_map.at(id_patch));
        field.insert(temp_pdat);
    });
}
