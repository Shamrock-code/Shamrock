// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindGhostLayerIndices.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/FindGhostLayerIndices.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>

namespace {

    template<class TgridVec>
    sham::DeviceBuffer<u32> get_ids_in_ghost(
        const sham::DeviceScheduler_ptr &dev_sched,
        sham::DeviceQueue &q,
        const sham::DeviceBuffer<TgridVec> &block_min,
        const sham::DeviceBuffer<TgridVec> &block_max,
        const shammath::AABB<TgridVec> &test_volume,
        u32 obj_cnt) {

        if (obj_cnt > 0) {
            // buffer of booleans to store result of the condition
            sham::DeviceBuffer<u32> mask(obj_cnt, dev_sched);

            sham::kernel_call(
                q,
                sham::MultiRef{block_min, block_max},
                sham::MultiRef{mask},
                obj_cnt,
                [test_volume](
                    u32 id,
                    const TgridVec *__restrict block_min,
                    const TgridVec *__restrict block_max,
                    u32 *__restrict is_in_ghost) {
                    is_in_ghost[id] = shammath::AABB<TgridVec>(block_min[id], block_max[id])
                                          .get_intersect(test_volume)
                                          .is_not_empty();
                });

            return shamalgs::stream_compact(dev_sched, mask, obj_cnt);
        } else {
            return sham::DeviceBuffer<u32>(0, dev_sched);
        }
    }
} // namespace

template<class TgridVec>
void shammodels::basegodunov::modules::FindGhostLayerIndices<TgridVec>::_impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box                 = edges.sim_box.value;
    auto &patch_data_layers       = edges.patch_data_layers;
    auto &ghost_layers_candidates = edges.ghost_layers_candidates;
    auto &patch_boxes             = edges.patch_boxes;

    // outputs
    auto &idx_in_ghost = edges.idx_in_ghost;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    auto paving_function = get_paving(mode, sim_box);

    auto &patch_data_layers_ref = patch_data_layers.get_const_refs();

    // map candidates to indexes in ghosts
    idx_in_ghost.buffers = ghost_layers_candidates.values.template map<sham::DeviceBuffer<u32>>(
        [&](u64 sender,
            u64 receiver,
            const GhostLayerCandidateInfos &infos) -> sham::DeviceBuffer<u32> {
            shamrock::patch::PatchDataLayer &sender_patch = patch_data_layers_ref.get(sender).get();

            PatchDataField<TgridVec> &block_min = sender_patch.get_field<TgridVec>(0);
            PatchDataField<TgridVec> &block_max = sender_patch.get_field<TgridVec>(1);

            auto brecv = patch_boxes.values.get(receiver);

            auto test_volume
                = paving_function.f_aabb_inv(brecv, infos.xoff, infos.yoff, infos.zoff);

            return get_ids_in_ghost(
                dev_sched,
                q,
                block_min.get_buf(),
                block_max.get_buf(),
                test_volume,
                sender_patch.get_obj_cnt());
        });
}

template<class TgridVec>
std::string shammodels::basegodunov::modules::FindGhostLayerIndices<TgridVec>::_impl_get_tex()
    const {
    return "TODO";
}

template class shammodels::basegodunov::modules::FindGhostLayerIndices<i64_3>;
