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
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/ExtractGhostLayer.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ExtractGhostLayer<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box                 = edges.sim_box.value;
    auto &patch_data_layers       = edges.patch_data_layers;
    auto &ghost_layers_candidates = edges.ghost_layers_candidates;
    auto &idx_in_ghost            = edges.idx_in_ghost;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    auto paving_function = get_paving(mode, sim_box);

    // extract the ghost layers
    auto idx_in_ghost_it     = idx_in_ghost.buffers.begin();
    auto ghost_layer_info_it = ghost_layers_candidates.values.begin();

    // iterate on both DDShared containers
    for (; idx_in_ghost_it != idx_in_ghost.buffers.end();
         ++idx_in_ghost_it, ++ghost_layer_info_it) {

        auto [sender, receiver] = idx_in_ghost_it->first;

        const sham::DeviceBuffer<u32> &sender_idx_in_ghost = idx_in_ghost_it->second;
        auto &sender_ghost_layer_info                      = ghost_layer_info_it->second;

        shamrock::patch::PatchDataLayer ghost_zone(ghost_layer_layout);

        // extract the actual data
        patch_data_layers.get(sender).append_subset_to(
            sender_idx_in_ghost, u32(sender_idx_in_ghost.get_size()), ghost_zone);

        ghost_layer.patchdatas.add_obj(sender, receiver, std::move(ghost_zone));
    }
}

template<class Tvec, class TgridVec>
std::string shammodels::basegodunov::modules::ExtractGhostLayer<Tvec, TgridVec>::_impl_get_tex() {
    return "TODO";
}

template class shammodels::basegodunov::modules::ExtractGhostLayer<f64_3, i64_3>;
