// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeRhoMean.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammodels/ramses/modules/ComputeRhoMean.hpp"
#include "shamrock/patch/PatchDataLayout.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    auto
    shammodels::basegodunov::modules::ComputeRhoMean<Tvec, TgridVec>::compute_rho_mean() -> Tscal {
        StackEntry stack_loc{};

        using namespace shamrock::patch;
        using namespace shamrock;
        using namespace shammath;

        shamrock::SchedulerUtility utility(scheduler());
        ComputeField<Tscal> mass = utility.make_compute_field<Tscal>("mass", AMRBlock::block_size);
        ComputeField<Tscal> vol  = utility.make_compute_field<Tscal>("vol", AMRBlock::block_size);

        // load layout info
        PatchDataLayout &pdl = scheduler().pdl;
        const u32 icell_min  = pdl.get_field_idx<TgridVec>("cell_min");
        const u32 icell_max  = pdl.get_field_idx<TgridVec>("cell_max");
        const u32 irho       = pdl.get_field_idx<Tscal>("rho");

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<TgridVec> &buf_block_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = pdat.get_field_buf_ref<TgridVec>(1);
            sham::DeviceBuffer<Tscal> &buf_rho          = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tscal> &buf_mass         = mass.get_buf_check(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_vol          = vol.get_buf_check(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(cur_p.id_patch)
                      .get()
                      .get_buf();

            sham::EventList depends_list;
            auto acc_mass      = buf_mass.get_write_access(depends_list);
            auto acc_vol       = buf_vol.get_write_access(depends_list);
            auto rho           = buf_rho.get_read_access(depends_list);
            auto acc_block_min = buf_block_min.get_read_access(depends_list);
            auto acc_block_max = buf_block_max.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                Tscal one_over_Nside = 1. / AMRBlock::Nside;
                Tscal dxfact         = solver_config.grid_coord_to_pos_fact;

                shambase::parralel_for(cgh, cell_count, "compute_mean_rho", [=](u64 gid) {
                    const u32 cell_global_id = (u32) gid;
                    const u32 bloc_id        = cell_global_id / AMRBlock::block_size;
                    const u32 cell_loc_id    = cell_global_id % AMRBlock::block_size;

                    TgridVec lower       = acc_block_min[bloc_id];
                    TgridVec upper       = acc_block_max[bloc_id];
                    Tvec lower_flt       = lower.template convert<Tscal>() * dxfact;
                    Tvec upper_flt       = upper.template convert<Tscal>() * dxfact;
                    Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;
                    Tscal dV = block_cell_size.x() * block_cell_size.y() * block_cell_size.z();

                    acc_vol[gid]  = dV;
                    acc_mass[gid] = rho[gid] * dV;
                });
            });
            buf_block_min.complete_event_state(e);
            buf_block_max.complete_event_state(e);
            buf_rho.complete_event_state(e);
            buf_mass.complete_event_state(e);
            buf_vol.complete_event_state(e);
        });

        Tscal rank_vol  = vol.compute_rank_sum();
        Tscal rank_mass = mass.compute_rank_sum();
        Tscal rho_tot   = shamalgs::collective::allreduce_sum(rank_mass);
        Tscal vol_tot   = shamalgs::collective::allreduce_sum(rank_vol);
        return rho_tot / vol_tot;
    }

} // namespace shammodels::basegodunov::modules
template class shammodels::basegodunov::modules::ComputeRhoMean<f64_3, i64_3>;
