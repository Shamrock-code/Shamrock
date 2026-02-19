// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FuseGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammodels/ramses/modules/FuseGhostLayer.hpp"

void shammodels::basegodunov::modules::FuseGhostLayer::_impl_evaluate_internal() {
    auto edges = get_edges();

    auto &ghost_layer       = edges.ghost_layer;
    auto &patch_data_layers = edges.patch_data_layers;

    ghost_layer.patchdatas.for_each(
        [&]([[maybe_unused]] u64 sender,
            u64 receiver,
            const shamrock::patch::PatchDataLayer &ghost_layer_element) {
            patch_data_layers.get(receiver).insert_elements(ghost_layer_element);
        });
}

std::string shammodels::basegodunov::modules::FuseGhostLayer::_impl_get_tex() const {
    return "TODO";
}
