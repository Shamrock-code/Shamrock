// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SRTimestepper.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Time integration implementation for SR-GSPH
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/gsph/physics/sr/SRPrimitiveRecovery.hpp"
#include "shammodels/gsph/physics/sr/SRTimestepper.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::physics::sr {

    template<class Tvec, template<class> class SPHKernel>
    void SRTimestepper<Tvec, SPHKernel>::do_predictor(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");

        const Tscal half_dt = dt / Tscal{2};

        if (storage.sr_initialized) {
            shamrock::solvergraph::Field<Tvec> &S_field   = *storage.S_momentum;
            shamrock::solvergraph::Field<Tscal> &e_field  = *storage.e_energy;
            shamrock::solvergraph::Field<Tvec> &dS_field  = *storage.dS_momentum;
            shamrock::solvergraph::Field<Tscal> &de_field = *storage.de_energy;

            // Half-step update of conserved variables
            scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                u32 cnt = pdat.get_obj_cnt();
                if (cnt == 0)
                    return;

                sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
                sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
                sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
                sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

                auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{buf_dS, buf_de},
                    sham::MultiRef{buf_S, buf_e},
                    cnt,
                    [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                        S[i] += dS[i] * half_dt;
                        e[i] += de[i] * half_dt;
                    });
            });

            // Recover primitives from updated conserved variables
            SRPrimitiveRecovery<Tvec, SPHKernel>::recover(storage, config, scheduler);
        }

        // Drift: x += v*dt
        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &xyz_field  = pdat.get_field<Tvec>(ixyz);
            auto &vxyz_field = pdat.get_field<Tvec>(ivxyz);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{vxyz_field.get_buf()},
                sham::MultiRef{xyz_field.get_buf()},
                cnt,
                [dt](u32 i, const Tvec *vxyz, Tvec *xyz) {
                    xyz[i] += vxyz[i] * dt;
                });
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    bool SRTimestepper<Tvec, SPHKernel>::apply_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        const Tscal half_dt = dt / Tscal{2};

        shamrock::solvergraph::Field<Tvec> &S_field   = *storage.S_momentum;
        shamrock::solvergraph::Field<Tscal> &e_field  = *storage.e_energy;
        shamrock::solvergraph::Field<Tvec> &dS_field  = *storage.dS_momentum;
        shamrock::solvergraph::Field<Tscal> &de_field = *storage.de_energy;

        scheduler.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            sham::DeviceBuffer<Tvec> &buf_S   = S_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_e  = e_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tvec> &buf_dS  = dS_field.get_buf(cur_p.id_patch);
            sham::DeviceBuffer<Tscal> &buf_de = de_field.get_buf(cur_p.id_patch);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{buf_dS, buf_de},
                sham::MultiRef{buf_S, buf_e},
                cnt,
                [half_dt](u32 i, const Tvec *dS, const Tscal *de, Tvec *S, Tscal *e) {
                    S[i] += dS[i] * half_dt;
                    e[i] += de[i] * half_dt;
                });
        });

        SRPrimitiveRecovery<Tvec, SPHKernel>::recover(storage, config, scheduler);

        return true;
    }

    template<class Tvec, template<class> class SPHKernel>
    void SRTimestepper<Tvec, SPHKernel>::prepare_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {
        // SR doesn't need to save old derivatives - predictor-corrector is symmetric
    }

    // Explicit instantiations
    using namespace shammath;

    template class SRTimestepper<sycl::vec<double, 3>, M4>;
    template class SRTimestepper<sycl::vec<double, 3>, M6>;
    template class SRTimestepper<sycl::vec<double, 3>, M8>;
    template class SRTimestepper<sycl::vec<double, 3>, C2>;
    template class SRTimestepper<sycl::vec<double, 3>, C4>;
    template class SRTimestepper<sycl::vec<double, 3>, C6>;
    template class SRTimestepper<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::sr
