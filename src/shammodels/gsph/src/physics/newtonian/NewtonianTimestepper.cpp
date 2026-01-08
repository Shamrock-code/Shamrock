// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NewtonianTimestepper.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Leapfrog time integration implementation for Newtonian GSPH
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/gsph/physics/newtonian/NewtonianTimestepper.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::physics::newtonian {

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianTimestepper<Tvec, SPHKernel>::do_predictor(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

        const bool has_uint = config.has_field_uint();
        const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>("uint") : 0;
        const u32 iduint    = has_uint ? pdl.get_field_idx<Tscal>("duint") : 0;

        Tscal half_dt = dt / Tscal{2};

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &xyz_field  = pdat.get_field<Tvec>(ixyz);
            auto &vxyz_field = pdat.get_field<Tvec>(ivxyz);
            auto &axyz_field = pdat.get_field<Tvec>(iaxyz);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            // Leapfrog KDK: first half-kick, then drift
            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{axyz_field.get_buf()},
                sham::MultiRef{xyz_field.get_buf(), vxyz_field.get_buf()},
                cnt,
                [half_dt, dt](u32 i, const Tvec *axyz, Tvec *xyz, Tvec *vxyz) {
                    // First kick: v += a*dt/2 (using OLD acceleration)
                    vxyz[i] += axyz[i] * half_dt;
                    // Drift: x += v*dt
                    xyz[i] += vxyz[i] * dt;
                });

            // Internal energy integration (if adiabatic EOS)
            if (has_uint) {
                auto &uint_field  = pdat.get_field<Tscal>(iuint);
                auto &duint_field = pdat.get_field<Tscal>(iduint);

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{duint_field.get_buf()},
                    sham::MultiRef{uint_field.get_buf()},
                    cnt,
                    [half_dt](u32 i, const Tscal *duint, Tscal *uint) {
                        uint[i] += duint[i] * half_dt;
                    });
            }
        });
    }

    template<class Tvec, template<class> class SPHKernel>
    bool NewtonianTimestepper<Tvec, SPHKernel>::apply_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler, Tscal dt) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

        Tscal half_dt = Tscal{0.5} * dt;

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &vxyz = pdat.get_field<Tvec>(ivxyz);
            auto &axyz = pdat.get_field<Tvec>(iaxyz);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{axyz.get_buf()},
                sham::MultiRef{vxyz.get_buf()},
                cnt,
                [half_dt](u32 i, const Tvec *axyz_new, Tvec *vxyz) {
                    vxyz[i] += half_dt * axyz_new[i];
                });
        });

        if (config.has_field_uint()) {
            const u32 iuint  = pdl.get_field_idx<Tscal>("uint");
            const u32 iduint = pdl.get_field_idx<Tscal>("duint");

            scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                u32 cnt = pdat.get_obj_cnt();
                if (cnt == 0)
                    return;

                auto &uint_field = pdat.get_field<Tscal>(iuint);
                auto &duint      = pdat.get_field<Tscal>(iduint);

                auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{duint.get_buf()},
                    sham::MultiRef{uint_field.get_buf()},
                    cnt,
                    [half_dt](u32 i, const Tscal *duint_new, Tscal *uint) {
                        uint[i] += half_dt * duint_new[i];
                    });
            });

            storage.old_duint.reset();
        }

        storage.old_axyz.reset();

        return true;
    }

    template<class Tvec, template<class> class SPHKernel>
    void NewtonianTimestepper<Tvec, SPHKernel>::prepare_corrector(
        Storage &storage, const Config &config, PatchScheduler &scheduler) {

        StackEntry stack_loc{};
        using namespace shamrock::patch;

        shamrock::SchedulerUtility utility(scheduler);
        PatchDataLayerLayout &pdl = scheduler.pdl();
        const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

        auto old_axyz = utility.make_compute_field<Tvec>("old_axyz", 1);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &axyz_field     = pdat.get_field<Tvec>(iaxyz);
            auto &old_axyz_field = old_axyz.get_field(p.id_patch);

            sham::kernel_call(
                dev_sched->get_queue(),
                sham::MultiRef{axyz_field.get_buf()},
                sham::MultiRef{old_axyz_field.get_buf()},
                cnt,
                [](u32 i, const Tvec *src, Tvec *dst) {
                    dst[i] = src[i];
                });
        });

        storage.old_axyz.set(std::move(old_axyz));

        if (config.has_field_uint()) {
            const u32 iduint = pdl.get_field_idx<Tscal>("duint");
            auto old_duint   = utility.make_compute_field<Tscal>("old_duint", 1);

            scheduler.for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                u32 cnt = pdat.get_obj_cnt();
                if (cnt == 0)
                    return;

                auto &duint_field     = pdat.get_field<Tscal>(iduint);
                auto &old_duint_field = old_duint.get_field(p.id_patch);

                sham::kernel_call(
                    dev_sched->get_queue(),
                    sham::MultiRef{duint_field.get_buf()},
                    sham::MultiRef{old_duint_field.get_buf()},
                    cnt,
                    [](u32 i, const Tscal *src, Tscal *dst) {
                        dst[i] = src[i];
                    });
            });

            storage.old_duint.set(std::move(old_duint));
        }
    }

    // Explicit instantiations
    using namespace shammath;

    template class NewtonianTimestepper<sycl::vec<double, 3>, M4>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, M6>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, M8>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, C2>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, C4>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, C6>;
    template class NewtonianTimestepper<sycl::vec<double, 3>, TGauss3>;

} // namespace shammodels::gsph::physics::newtonian
