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
 * @file AnalysisAngularMomentum.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisAngularMomentum class
 *
 */

#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammodels/sph/Model.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisAngularMomentum {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisAngularMomentum(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_angular_momentum() -> Tvec {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ivxyz   = sched.pdl_old().template get_field_idx<Tvec>("vxyz");
            const u32 ixyz    = sched.pdl_old().template get_field_idx<Tvec>("xyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tvec angular_momentum = {};

            sham::DeviceBuffer<Tvec> angular_momentum_part(0, dev_sched_ptr);

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    u32 len = pdat.get_obj_cnt();

                    angular_momentum_part.resize(len);

                    sham::DeviceBuffer<Tvec> &xyz_buf  = pdat.get_field_buf_ref<Tvec>(ixyz);
                    sham::DeviceBuffer<Tvec> &vxyz_buf = pdat.get_field_buf_ref<Tvec>(ivxyz);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{xyz_buf, vxyz_buf},
                        sham::MultiRef{angular_momentum_part},
                        len,
                        [pmass](
                            u32 i,
                            const Tvec *__restrict xyz,
                            const Tvec *__restrict vxyz,
                            Tvec *__restrict angular_momentum_part) {
                            angular_momentum_part[i] = pmass * sycl::cross(xyz[i], vxyz[i]);
                        });

                    angular_momentum
                        += shamalgs::primitives::sum(dev_sched_ptr, angular_momentum_part, 0, len);
                });

            Tvec tot_angular_momentum = shamalgs::collective::allreduce_sum(angular_momentum);

            if (!solver.storage.sinks.is_empty()) {
                for (auto &sink : solver.storage.sinks.get()) {
                    tot_angular_momentum
                        += sink.mass * sycl::cross(sink.pos, sink.velocity) + sink.angular_momentum;
                }
            }

            return tot_angular_momentum;
        }
    };
} // namespace shammodels::sph::modules
