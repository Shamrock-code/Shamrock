// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DiffOperator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/zeus/modules/DiffOperator.hpp"
#include "shammodels/zeus/NeighFaceList.hpp"
#include "shammodels/zeus/modules/FaceFlagger.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamtree/TreeTraversal.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::DiffOperator<Tvec, TgridVec>::compute_gradu() {

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    shamrock::SchedulerUtility utility(scheduler());
    storage.gradu.set(utility.make_compute_field<Tvec>("gradu", 3, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayerLayout &ghost_layout
        = shambase::get_check_ref(storage.ghost_layout.get());
    u32 ivel_interf = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchDataLayer &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        shammodels::zeus::NeighFaceList<Tvec> &face_lists
            = storage.face_lists.get().get(p.id_patch);

        OrientedNeighFaceList<Tvec> &face_xm = face_lists.xm();
        OrientedNeighFaceList<Tvec> &face_ym = face_lists.ym();
        OrientedNeighFaceList<Tvec> &face_zm = face_lists.zm();

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        sham::DeviceBuffer<Tvec> &buf_vel    = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);
        sham::DeviceBuffer<Tvec> &buf_grad_u = storage.gradu.get().get_buf_check(p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;

        auto cell_min = buf_cell_min.get_read_access(depends_list);
        auto cell_max = buf_cell_max.get_read_access(depends_list);

        auto vel    = buf_vel.get_read_access(depends_list);
        auto grad_u = buf_grad_u.get_write_access(depends_list);

        auto faces_xm_ptr = face_xm.neigh_info.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            tree::ObjectCacheIterator faces_xm(faces_xm_ptr);

            shambase::parallel_for(cgh, pdat.get_obj_cnt(), "subsetp1", [=](u64 id_a) {
                Tvec cell2_a = (cell_min[id_a] + cell_max[id_a]).template convert<Tscal>()
                               * coord_conv_fact * 0.5f;

                Tvec sum_grad_ux = {};

                // looks like it's on the double preicision roofline there is
                // nothing to optimize here turn around
                faces_xm.for_each_object(id_a, [&](u32 id_b) {
                    Tvec cell2_b = (cell_min[id_b] + cell_max[id_b]).template convert<Tscal>()
                                   * coord_conv_fact * 0.5f;

                    Tvec n        = {-1, 0, 0};
                    Tscal dr_proj = sycl::dot(cell2_b - cell2_a, n);

                    Tvec drm1_n = n / dr_proj;

                    // buf_grad_u += drm1_n * rho[id_b];
                });

                // grad_p[id_a] = -buf_grad_u;
            });
        });

        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);
        buf_vel.complete_event_state(e);
        buf_grad_u.complete_event_state(e);

        sham::EventList resulting_events;
        resulting_events.add_event(e);
        face_xm.neigh_info.complete_event_state(resulting_events);
    });
}

template class shammodels::zeus::modules::DiffOperator<f64_3, i64_3>;
