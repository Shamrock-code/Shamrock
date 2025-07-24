// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputePressure.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shammodels/zeus/modules/ComputePressure.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::ComputePressure<Tvec, TgridVec>::compute_p() {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    using Block = typename Config::AMRBlock;

    storage.pressure.set(
        utility.make_compute_field<Tscal>("pressure", Block::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        }));

    ComputeField<Tscal> &pressure_field = storage.pressure.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        PatchDataField<Tscal> &press = storage.pressure.get().get_field(p.id_patch);

        sham::DeviceBuffer<Tscal> &buf_p    = pressure_field.get_buf_check(p.id_patch);
        sham::DeviceBuffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sham::DeviceBuffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;
        auto rho      = buf_rho.get_read_access(depends_list);
        auto eint     = buf_eint.get_read_access(depends_list);
        auto pressure = buf_p.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal gamma = solver_config.eos_gamma;

            shambase::parallel_for(
                cgh, mpdat.total_elements * Block::block_size, "compute pressure", [=](u64 id_a) {
                    pressure[id_a] = (gamma - 1) /** rho[id_a]*/ * eint[id_a];
                });
        });

        buf_rho.complete_event_state(e);
        buf_eint.complete_event_state(e);
        buf_p.complete_event_state(e);
    });
}

template class shammodels::zeus::modules::ComputePressure<f64_3, i64_3>;
