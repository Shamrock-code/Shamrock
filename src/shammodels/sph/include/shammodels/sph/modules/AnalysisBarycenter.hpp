// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AnalysisBarycenter.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisBarycenter class with one method AnalysisBarycenter.get_barycenter()
 *
 */

#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammodels/sph/Model.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisBarycenter {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisBarycenter(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        struct result {
            Tvec barycenter;
            Tscal mass_disc;
        };
        ////////////////////////////////////////////////////////////////////////////////////////////
        /////// setup function
        ////////////////////////////////////////////////////////////////////////////////////////////
        auto get_barycenter() -> result {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ixyz    = sched.pdl_old().template get_field_idx<Tvec>("xyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tvec barycenter = {0, 0, 0}; // Not really barycenter per se but Mdisc * barycenter
            Tscal mass_disc = 0;

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    // auto &xyz_buf = pdat.get_field_buf_ref<Tvec>(ixyz);
                    // u32 len       = pdat.get_obj_cnt();
                    // {
                    //     auto acc_xyz = xyz_buf.copy_to_stdvec();
                    //     for (u32 i = 0; i < len; i++) {
                    //         Tvec xyz = acc_xyz[i];

                    //         barycenter += pmass * xyz;
                    //         mass_disc += pmass;
                    //     }
                    // }
                    u32 len = pdat.get_obj_cnt();
                    mass_disc += pmass * len;

                    sham::DeviceBuffer<Tvec> pm(len, dev_sched_ptr);
                    sham::DeviceBuffer<Tvec> &xyz_buf = pdat.get_field_buf_ref<Tvec>(ixyz);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{xyz_buf},
                        sham::MultiRef{pm},
                        len,
                        [pmass](u32 i, const Tvec *__restrict xyz, Tvec *__restrict pm) {
                            pm[i] = pmass * xyz[i];
                        });
                    barycenter += shamalgs::primitives::sum(dev_sched_ptr, pm, 0, len);
                });

            Tvec tot_barycenter = shamalgs::collective::allreduce_sum(barycenter);
            Tscal tot_mass_disc = shamalgs::collective::allreduce_sum(mass_disc);
            Tscal tot_mass      = tot_mass_disc;

            if (!solver.storage.sinks.is_empty()) {
                for (auto &sink : solver.storage.sinks.get()) {
                    Tvec star_xyz   = sink.pos;
                    Tscal star_mass = sink.mass;

                    tot_barycenter += star_xyz * star_mass;
                    tot_mass += star_mass;
                }
            }
            tot_barycenter /= tot_mass;
            return result{tot_barycenter, tot_mass_disc};
        }
    };
} // namespace shammodels::sph::modules
