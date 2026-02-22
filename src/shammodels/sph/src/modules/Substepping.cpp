// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Substepping.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/Substepping.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/reduction.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::Substepping<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();
    using namespace shamrock;
    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
    auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
    SchedulerUtility utility(sched);

    auto edges = get_edges();

    Tscal dt_sph   = edges.dt_sph.data;
    Tscal dt_force = edges.dt_force.data;

    int n_substeps  = 0;
    bool done       = false;
    Tscal t_current = edges.current_time.data;
    Tscal t_end     = t_current + dt_sph;

    while (t_current < t_end && !done) {

        // kick
        // accrete particles+ update sinks
        // @@ sink_update.kick(dt_force / 2, true); // only external acc)
        shamlog_debug_ln("Substep", "kick");
        shambase::get_check_ref(do_foward_euler_vxyz_ptr).evaluate(); // vxyz
        shambase::get_check_ref(do_foward_euler_u_ptr).evaluate();    // u

        // @@ sink_update.accrete_particles();
        // @@ ext_forces.point_mass_accrete_particles();

        // drift + drift sinks
        shamlog_debug_ln("Substep", "drift");
        shambase::get_check_ref(do_foward_euler_xyz_ptr).evaluate(); // xyz
        // @@ sink_update.drift(dt_force / 2);
        // sink_update.accrete_particles();
        // ext_forces.point_mass_accrete_particles();

        // compute short range accelerations
        // reset axyz_ext to zero ?
        shammodels::sph::modules::ExternalForces<Tvec, SPHKernel> ext_forces(
            ctx, solver_config, storage);
        // axyz_ext.reset();
        ext_forces.compute_ext_forces_indep_v(); // @@@ pb here ?

        // kick
        shamlog_debug_ln("Substep", "kick");
        // sink_update.kick(dt_force / 2, true);
        shambase::get_check_ref(do_foward_euler_vxyz_ptr).evaluate(); // vxyz
        shambase::get_check_ref(do_foward_euler_u_ptr).evaluate();    // u
        // sink_update.accrete_particles();
        // ext_forces.point_mass_accrete_particles();

        logger::raw_ln("Kick 2 passed, now updating dt");
        // update dtforce
        Tscal sink_sink_cfl = shambase::get_infty<Tscal>();
        if (!storage.sinks.is_empty()) {
            // sink sink CFL
            Tscal G = solver_config.get_constant_G();
            Tscal C_force
                = solver_config.cfl_config.cfl_force * solver_config.time_state.cfl_multiplier;
            Tscal eta_phi                                    = solver_config.cfl_config.eta_sink;
            std::vector<sph::SinkParticle<Tvec>> &sink_parts = storage.sinks.get();

            for (u32 i = 0; i < sink_parts.size(); i++) {
                sph::SinkParticle<Tvec> &s_i = sink_parts[i];
                Tscal sink_sink_cfl_i        = shambase::get_infty<Tscal>();
                Tvec f_i                     = s_i.ext_acceleration;
                Tscal grad_phi_i_sq          = sham::dot(f_i, f_i); // m^2.s^-4
                if (grad_phi_i_sq == 0) {
                    continue;
                }

                for (u32 j = 0; j < sink_parts.size(); j++) {
                    sph::SinkParticle<Tvec> &s_j = sink_parts[j];

                    if (i == j) {
                        continue;
                    }

                    Tvec rij       = s_i.pos - s_j.pos;
                    Tscal rij_scal = sycl::length(rij);

                    Tscal phi_ij  = G * s_j.mass / rij_scal;                 // J / kg = m^2.s^-2
                    Tscal term_ij = sham::abs(phi_ij) / grad_phi_i_sq;       // s^2
                    Tscal dt_ij   = C_force * eta_phi * sycl::sqrt(term_ij); // s

                    sink_sink_cfl_i = sham::min(sink_sink_cfl_i, dt_ij);
                }

                sink_sink_cfl = sham::min(sink_sink_cfl, sink_sink_cfl_i);
            }

            sink_sink_cfl = shamalgs::collective::allreduce_min(sink_sink_cfl);
        }

        shamrock::ComputeField<Tscal> dt_force_arr
            = utility.make_compute_field<Tscal>("dt_force_arr", 1);

        const u32 iaxyz     = sched.pdl().template get_field_idx<Tvec>("axyz");
        const u32 iaxyz_ext = sched.pdl().template get_field_idx<Tvec>("axyz_ext");
        const u32 ihpart    = sched.pdl().template get_field_idx<Tvec>("hpart");
        logger::raw_ln("before loop on patches");
        sched.for_each_patchdata_nonempty([&](shamrock::patch::Patch cur_p,
                                              shamrock::patch::PatchDataLayer &pdat) {
            sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field<Tvec>(iaxyz_ext).get_buf();
            sham::DeviceBuffer<Tscal> &buf_hpart   = pdat.get_field<Tscal>(ihpart).get_buf();
            sham::DeviceBuffer<Tscal> &buf_dt_force_arr
                = dt_force_arr.get_buf_check(cur_p.id_patch);

            sham::DeviceQueue &q = shambase::get_check_ref(dev_sched_ptr).get_queue();
            sham::EventList depends_list;

            auto hpart        = buf_hpart.get_read_access(depends_list);
            auto aext         = buf_axyz_ext.get_read_access(depends_list);
            auto dt_force_arr = buf_dt_force_arr.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                Tscal C_force
                    = solver_config.cfl_config.cfl_force * solver_config.time_state.cfl_multiplier;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    Tscal h_a        = hpart[item];
                    Tscal abs_aext_a = sycl::length(aext[item]);

                    Tscal dt_f = C_force * sycl::sqrt(h_a / abs_aext_a);

                    dt_force_arr[item] = dt_f;
                });
            });

            buf_hpart.complete_event_state(e);
            buf_axyz_ext.complete_event_state(e);
            buf_dt_force_arr.complete_event_state(e);
        });

        logger::raw_ln("after loop on patches");

        Tscal rank_dt        = dt_force_arr.compute_rank_min();
        Tscal next_force_cfl = shamalgs::collective::allreduce_min(rank_dt);
        logger::raw_ln("all reduce passed");
        next_force_cfl = sycl::min(next_force_cfl, sink_sink_cfl);
        logger::raw_ln("min passed");
        solver_config.set_next_dt_force(next_force_cfl);

        logger::raw_ln("after asigned next dt force");
        // update time
        // solver_config.set_time(t_current + dt_force);
        // t_current = solver_config.get_time();
        t_current = t_current + dt_force;
        n_substeps++;

        bool last_step = dt_force >= dt_sph;
        if (last_step) {
            // last step
            done = true;
        } else {
            if (t_current + dt_force > t_end) {
                dt_force  = t_end - t_current;
                last_step = true;
            }
        }
    }
}

using namespace shammath;
template class shammodels::sph::modules::Substepping<f64_3, M4>;
template class shammodels::sph::modules::Substepping<f64_3, M6>;
template class shammodels::sph::modules::Substepping<f64_3, M8>;

template class shammodels::sph::modules::Substepping<f64_3, C2>;
template class shammodels::sph::modules::Substepping<f64_3, C4>;
template class shammodels::sph::modules::Substepping<f64_3, C6>;
