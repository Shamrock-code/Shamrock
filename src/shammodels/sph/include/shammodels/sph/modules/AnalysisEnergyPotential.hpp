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
 * @file AnalysisEnergyPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisBarycenter class with one method AnalysisBarycenter.get_baycenter()
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
    class AnalysisEnergyPotential {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisEnergyPotential(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_potential_energy() -> Tscal {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ixyz    = sched.pdl().template get_field_idx<Tvec>("xyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tscal epot = 0;

            if (!solver.storage.sinks.is_empty()) {
                for (auto &sink : solver.storage.sinks.get()) {
                    sched.for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                          shamrock::patch::PatchDataLayer &pdat) {
                        u32 len = pdat.get_obj_cnt();

                        sham::DeviceBuffer<Tscal> epot_part(len, dev_sched_ptr);
                        sham::DeviceBuffer<Tvec> &xyz_buf = pdat.get_field_buf_ref<Tvec>(ixyz);

                        sham::kernel_call(
                            q,
                            sham::MultiRef{xyz_buf},
                            sham::MultiRef{epot_part},
                            len,
                            [pmass,
                             smass    = sink.mass,
                             G        = solver.solver_config.get_constant_G(),
                             sink_pos = sink.pos](
                                u32 i, const Tvec *__restrict xyz, Tscal *__restrict epot_part) {
                                epot_part[i] = -pmass * G * smass / sycl::length(xyz[i] - sink_pos);
                            });

                        epot += shamalgs::primitives::sum(dev_sched_ptr, epot_part, 0, len);
                    });
                }

                Tscal G = solver.solver_config.get_constant_G();

                for (auto &sink1 : solver.storage.sinks.get()) {
                    for (auto &sink2 : solver.storage.sinks.get()) {
                        Tvec delta = sink1.pos - sink2.pos;
                        Tscal d    = sycl::length(delta);

                        if (d > 1e-6 * (sink1.accretion_radius + sink2.accretion_radius)) {
                            epot += -G * sink1.mass * sink2.mass / d;
                        }
                    }
                }
            }

            Tscal tot_epot = shamalgs::collective::allreduce_sum(epot);

            return tot_epot;
        }
    };
} // namespace shammodels::sph::modules
