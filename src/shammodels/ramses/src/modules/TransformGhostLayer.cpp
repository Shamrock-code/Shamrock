// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TransformGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/TransformGhostLayer.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box                 = edges.sim_box.value;
    auto &ghost_layers_candidates = edges.ghost_layers_candidates;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    auto paving_function = get_paving(mode, sim_box);

    // get the block min and max field
    u32 iblock_min = ghost_layer_layout->get_field_idx<TgridVec>("cell_min");
    u32 iblock_max = ghost_layer_layout->get_field_idx<TgridVec>("cell_max");

    // extract the ghost layers
    auto ghost_layer_it      = ghost_layer.patchdatas.begin();
    auto ghost_layer_info_it = ghost_layers_candidates.values.begin();

    if (ghost_layer.patchdatas.get_element_count()
        != ghost_layers_candidates.values.get_element_count()) {
        shambase::throw_with_loc<std::runtime_error>(shambase::format(
            "ghost_layer.patchdatas.get_element_count() != "
            "ghost_layers_candidates.values.get_element_count()\n "
            "ghost_layer.patchdatas.get_element_count(): {}\n"
            "ghost_layers_candidates.values.get_element_count(): {}",
            ghost_layer.patchdatas.get_element_count(),
            ghost_layers_candidates.values.get_element_count()));
    }

    // iterate on both DDShared containers
    for (; ghost_layer_it != ghost_layer.patchdatas.end();
         ++ghost_layer_it, ++ghost_layer_info_it) {

        auto [sender, receiver] = ghost_layer_it->first;

        shamrock::patch::PatchDataLayer &ghost_layer_element = ghost_layer_it->second;
        auto &sender_ghost_layer_info                        = ghost_layer_info_it->second;

        auto &block_min_buf = ghost_layer_element.get_field<TgridVec>(iblock_min).get_buf();
        auto &block_max_buf = ghost_layer_element.get_field<TgridVec>(iblock_max).get_buf();

        // transform the block min and max
        sham::kernel_call(
            q,
            sham::MultiRef{},
            sham::MultiRef{block_min_buf, block_max_buf},
            ghost_layer_element.get_obj_cnt(),
            [paving_function,
             xoff = sender_ghost_layer_info.xoff,
             yoff = sender_ghost_layer_info.yoff,
             zoff = sender_ghost_layer_info.zoff](
                u32 i, TgridVec *__restrict block_min, TgridVec *__restrict block_max) {
                shammath::AABB<TgridVec> block_box = {block_min[i], block_max[i]};

                block_box = paving_function.f_aabb(block_box, xoff, yoff, zoff);

                block_min[i] = block_box.lower;
                block_max[i] = block_box.upper;
            });

        // do not forget that while we have transformed the ghost layer block bound we did not
        // transform the ghost layer data Especially if the paving is reflexive a permutation needs
        // to be applied to the ghost layer data

        // TODO
    }
}

template<class Tvec, class TgridVec>
std::string shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>::_impl_get_tex() {
    return "TODO";
}

template class shammodels::basegodunov::modules::TransformGhostLayer<f64_3, i64_3>;
