// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the ExtractGhostLayer solver graph node.
 */

#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"

void shammodels::basegodunov::modules::ExtractGhostLayer::_impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    // inputs
    auto &patch_data_layers = edges.patch_data_layers;
    auto &idx_in_ghost      = edges.idx_in_ghost;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    // iterate on buffer storing indexes in ghost layer
    for (const auto &[key, sender_idx_in_ghost] : idx_in_ghost.buffers) {
        auto [sender, receiver] = key;

        shamrock::patch::PatchDataLayer ghost_zone(ghost_layer_layout);

        // extract the actual data
        patch_data_layers.get(sender).append_subset_to(
            sender_idx_in_ghost, u32(sender_idx_in_ghost.get_size()), ghost_zone);

        ghost_layer.patchdatas.add_obj(sender, receiver, std::move(ghost_zone));
    }
}
