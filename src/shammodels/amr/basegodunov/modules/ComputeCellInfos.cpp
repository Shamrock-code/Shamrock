// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCellInfos.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "ComputeCellInfos.hpp"
#include "shammodels/common/amr/AMRCellInfos.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeCellInfos<Tvec, TgridVec>::compute_aabb() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    logger::debug_ln("AMR grid", "compute block/cell infos");

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tscal> block_cell_sizes
        = utility.make_compute_field<Tscal>("aabb cell size", 1, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });
    shamrock::ComputeField<Tvec> cell0block_aabb_lower
        = utility.make_compute_field<Tvec>("aabb cell lower", 1, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::EventList depends_list;

        auto acc_block_min = buf_block_min.get_read_access(depends_list);
        auto acc_block_max = buf_block_max.get_read_access(depends_list);
        auto bsize         = block_cell_sizes.get_buf(id).get_write_access(depends_list);
        auto aabb_lower    = cell0block_aabb_lower.get_buf(id).get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal one_over_Nside = 1. / AMRBlock::Nside;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;

            shambase::parralel_for(cgh, mpdat.total_elements, "compute cell infos", [=](u32 gid) {
                TgridVec lower = acc_block_min[gid];
                TgridVec upper = acc_block_max[gid];

                Tvec lower_flt = lower.template convert<Tscal>() * dxfact;
                Tvec upper_flt = upper.template convert<Tscal>() * dxfact;

                Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

                Tscal res = block_cell_size.x();

                bsize[gid]      = res;
                aabb_lower[gid] = lower_flt;
            });
        });

        buf_block_min.complete_event_state(e);
        buf_block_max.complete_event_state(e);
        block_cell_sizes.get_buf(id).complete_event_state(e);
        cell0block_aabb_lower.get_buf(id).complete_event_state(e);
    });

    storage.cell_infos.set(
        CellInfos<Tvec, TgridVec>{std::move(block_cell_sizes), std::move(cell0block_aabb_lower)});
}

template class shammodels::basegodunov::modules::ComputeCellInfos<f64_3, i64_3>;
