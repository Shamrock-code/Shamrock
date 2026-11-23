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

        sham::kernel_call(
            q,
            sham::MultiRef{face_xm.neigh_info, buf_cell_min, buf_cell_max, buf_vel},
            sham::MultiRef{buf_grad_u},
            pdat.get_obj_cnt(),
            [coord_conv_fact](
                u32 id_a,
                auto faces_xm_ptr,
                const TgridVec *__restrict cell_min,
                const TgridVec *__restrict cell_max,
                const Tvec *__restrict vel,
                Tvec *__restrict grad_u) {
                tree::ObjectCacheIterator faces_xm(faces_xm_ptr);

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
}

template class shammodels::zeus::modules::DiffOperator<f64_3, i64_3>;
