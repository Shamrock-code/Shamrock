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
 * @file AnalysisDustMass.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisDustMass class
 *
 */

#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisDustMass {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = Solver<Tvec, SPHKernel>;

        using Kernel = SPHKernel<Tscal>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisDustMass(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_dust_mass() -> std::vector<Tscal> {

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            u64 ndust = solver.solver_config.dust_config.get_dust_nvar();

            const u32 ihpart = sched.pdl_old().template get_field_idx<Tscal>("hpart");
            const u32 is_j   = sched.pdl_old().template get_field_idx<Tscal>("s_j");

            Tscal pmass = solver.solver_config.gpart_mass;

            std::vector<Tscal> dust_mass(ndust, 0.0);

            sham::DeviceBuffer<Tscal> dust_mass_j_part(0, dev_sched_ptr);

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    u32 len = pdat.get_obj_cnt();

                    dust_mass_j_part.resize(len);

                    sham::DeviceBuffer<Tscal> &hpart_buf = pdat.get_field_buf_ref<Tscal>(ihpart);
                    sham::DeviceBuffer<Tscal> &s_j_buf   = pdat.get_field_buf_ref<Tscal>(is_j);

                    for (u32 jdust = 0; jdust < ndust; jdust++) {
                        sham::kernel_call(
                            q,
                            sham::MultiRef{hpart_buf, s_j_buf},
                            sham::MultiRef{dust_mass_j_part},
                            len,
                            [pmass, jdust, ndust](
                                u32 i,
                                const Tscal *__restrict hpart,
                                const Tscal *__restrict s_j,
                                Tscal *__restrict dust_mass_j_part) {
                                Tscal h_a        = hpart[i];
                                Tscal rho_a      = shamrock::sph::rho_h(pmass, h_a, Kernel::hfactd);
                                Tscal s_ja       = s_j[i * ndust + jdust];
                                Tscal epsilon_ja = s_ja * s_ja / rho_a;

                                dust_mass_j_part[i] = pmass * epsilon_ja;
                            });

                        dust_mass[jdust]
                            += shamalgs::primitives::sum(dev_sched_ptr, dust_mass_j_part, 0, len);
                    }
                });

            for (u32 jdust = 0; jdust < ndust; jdust++) {
                dust_mass[jdust] = shamalgs::collective::allreduce_sum(dust_mass[jdust]);
            }

            return dust_mass;
        }
    };
} // namespace shammodels::sph::modules
