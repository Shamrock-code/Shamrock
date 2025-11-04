// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AnalysisEnergyKinetic.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisEnergyKinetic class with one method AnalysisEnergyKinetic.get_kinetic_energy()
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
    class AnalysisEnergyKinetic {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = ::shammodels::sph::Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisEnergyKinetic(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_kinetic_energy() -> Tscal {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ivxyz   = sched.pdl().template get_field_idx<Tvec>("vxyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tscal ekin = 0;

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    u32 len = pdat.get_obj_cnt();

                    sham::DeviceBuffer<Tscal> ekin_part(len, dev_sched_ptr);
                    sham::DeviceBuffer<Tvec> &vxyz_buf = pdat.get_field_buf_ref<Tvec>(ivxyz);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{vxyz_buf},
                        sham::MultiRef{ekin_part},
                        len,
                        [pmass](u32 i, const Tvec *__restrict vxyz, Tscal *__restrict ekin_part) {
                            ekin_part[i] = Tscal{0.5} * pmass * sham::dot(vxyz[i], vxyz[i]);
                        });

                    ekin += shamalgs::primitives::sum(dev_sched_ptr, ekin_part, 0, len);
                });

            Tscal tot_ekin = shamalgs::collective::allreduce_sum(ekin);

            if (!solver.storage.sinks.is_empty()) {
                for (auto &sink : solver.storage.sinks.get()) {
                    tot_ekin += Tscal{0.5} * sink.mass * sham::dot(sink.velocity, sink.velocity);
                }
            }

            return tot_ekin;
        }
    };
} // namespace shammodels::sph::modules
