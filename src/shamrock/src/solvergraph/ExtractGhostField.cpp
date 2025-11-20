// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostField.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"

namespace shamrock::solvergraph {
    template<class T>
    void shamrock::solvergraph::ExtractGhostField<T>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        auto &orig_data_fields = edges.original_fields;
        auto &idx_in_ghost     = edges.idx_in_ghots;
        auto &ghost_fields     = edges.ghost_fields;

        ghost_fields.patchdata_fields.reset(); // To make sure there is no residual of last timestep

        for (const auto &[key, sender_idx_in_ghost] : idx_in_ghost.buffers) {
            auto [sender, receiver] = key;
            auto field_name         = orig_data_fields.get_field(sender).get_name();
            auto nvar               = orig_data_fields.get_field(sender).get_nvar();
            PatchDataField<T> gz_field(field_name, nvar);

            orig_data_fields.get_field(sender).append_subset_to(
                sender_idx_in_ghost, (u32) sender_idx_in_ghost.get_size(), gz_field);

            ghost_fields.patchdata_fields.add_obj(sender, receiver, std::move(gz_field));
        }
    }

    template class ExtractGhostField<f64>;
    template class ExtractGhostField<f64_3>;

} // namespace shamrock::solvergraph
