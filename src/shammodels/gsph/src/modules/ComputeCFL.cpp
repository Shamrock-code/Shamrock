// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCFL.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief CFL timestep constraint computation implementation
 */

#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shammodels/gsph/modules/ComputeCFL.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    typename ComputeCFL<Tvec, SPHKernel>::Tscal ComputeCFL<Tvec, SPHKernel>::compute() {
        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ihpart          = pdl.get_field_idx<Tscal>("hpart");

        shamrock::solvergraph::Field<Tscal> &soundspeed_field
            = shambase::get_check_ref(storage.soundspeed);

        Tscal C_cour = solver_config.cfl_config.cfl_cour;

        shamrock::SchedulerUtility utility(scheduler());
        ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", 1);

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 cnt = pdat.get_obj_cnt();
            if (cnt == 0)
                return;

            auto &buf_hpart  = pdat.get_field_buf_ref<Tscal>(ihpart);
            auto &buf_cs     = soundspeed_field.get_field(cur_p.id_patch).get_buf();
            auto &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

            sham::DeviceQueue &q = dev_sched->get_queue();
            sham::EventList depends_list;

            auto hpart      = buf_hpart.get_read_access(depends_list);
            auto cs         = buf_cs.get_read_access(depends_list);
            auto cfl_dt_acc = cfl_dt_buf.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cnt, "gsph_compute_cfl_dt", [=](u64 gid) {
                    u32 i = (u32) gid;

                    Tscal h_i  = hpart[i];
                    Tscal cs_i = cs[i];

                    // FAIL FAST: Detect bad storage values
                    if (!sycl::isfinite(h_i) || h_i <= Tscal(0)) {
                        printf("CFL ERROR: particle %u has invalid h=%.6e\n", i, (double) h_i);
                    }
                    if (!sycl::isfinite(cs_i) || cs_i <= Tscal(0)) {
                        printf(
                            "CFL ERROR: particle %u has invalid cs=%.6e (h=%.6e)\n",
                            i,
                            (double) cs_i,
                            (double) h_i);
                    }

                    // CFL: Only use Courant condition (h/cs)
                    Tscal dt_min = C_cour * h_i / cs_i;

                    if (!sycl::isfinite(dt_min) || dt_min <= Tscal(0)) {
                        printf(
                            "CFL ERROR: particle %u has invalid dt=%.6e (h=%.6e, cs=%.6e)\n",
                            i,
                            (double) dt_min,
                            (double) h_i,
                            (double) cs_i);
                        dt_min = Tscal(1e-10); // Continue to expose more errors
                    }

                    cfl_dt_acc[i] = dt_min;
                });
            });

            buf_hpart.complete_event_state(e);
            buf_cs.complete_event_state(e);
            cfl_dt_buf.complete_event_state(e);
        });

        Tscal rank_dt = cfl_dt.compute_rank_min();

        if (!std::isfinite(rank_dt) || rank_dt <= Tscal(0)) {
            shambase::throw_with_loc<std::runtime_error>(
                "CFL FAIL: rank_dt is invalid (" + std::to_string(rank_dt)
                + "). Check h and cs fields for NaN/Inf/zero values.");
        }

        Tscal global_min_dt = shamalgs::collective::allreduce_min(rank_dt);

        if (!std::isfinite(global_min_dt) || global_min_dt <= Tscal(0)) {
            shambase::throw_with_loc<std::runtime_error>(
                "CFL FAIL: global_min_dt is invalid (" + std::to_string(global_min_dt)
                + "). Storage fields may contain NaN/Inf values.");
        }

        return global_min_dt;
    }

    // Explicit instantiations
    template class ComputeCFL<f64_3, shammath::M4>;
    template class ComputeCFL<f64_3, shammath::M6>;
    template class ComputeCFL<f64_3, shammath::M8>;
    template class ComputeCFL<f64_3, shammath::C2>;
    template class ComputeCFL<f64_3, shammath::C4>;
    template class ComputeCFL<f64_3, shammath::C6>;
    template class ComputeCFL<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules
