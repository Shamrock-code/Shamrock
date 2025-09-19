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
 * @file AnalysisTotalMomentum.hpp
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
    class AnalysisTotalMomentum {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisTotalMomentum(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_total_momentum() -> Tvec {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ivxyz   = sched.pdl().template get_field_idx<Tvec>("vxyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tvec total_momentum = {};

            sched.for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                  shamrock::patch::PatchDataLayer &pdat) {
                u32 len = pdat.get_obj_cnt();

                sham::DeviceBuffer<Tvec> total_momentum_part(len, dev_sched_ptr);
                sham::DeviceBuffer<Tvec> &vxyz_buf = pdat.get_field_buf_ref<Tvec>(ivxyz);

                sham::kernel_call(
                    q,
                    sham::MultiRef{vxyz_buf},
                    sham::MultiRef{total_momentum_part},
                    len,
                    [pmass](
                        u32 i, const Tvec *__restrict vxyz, Tvec *__restrict total_momentum_part) {
                        total_momentum_part[i] = pmass * vxyz[i];
                    });

                total_momentum
                    += shamalgs::primitives::sum(dev_sched_ptr, total_momentum_part, 0, len);
            });

            Tvec tot_total_momentum = shamalgs::collective::allreduce_sum(total_momentum);

            if (!solver.storage.sinks.is_empty()) {
                for (auto &sink : solver.storage.sinks.get()) {
                    tot_total_momentum += sink.mass * sink.velocity;
                }
            }

            return tot_total_momentum;
        }
    };
} // namespace shammodels::sph::modules
