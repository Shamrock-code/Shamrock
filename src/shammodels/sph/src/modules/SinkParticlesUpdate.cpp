// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesUpdate.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/narrowing.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include <shambackends/sycl.hpp>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::accrete_particles(Tscal dt) {
    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    if (storage.sinks.is_empty()) {
        return;
    }

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    std::vector<Sink> &sink_parts = storage.sinks.get();

    u32 sink_id        = 0;
    bool had_accretion = false;
    std::string log    = "sink accretion :";

    struct AccretionFlagBufs {
        sham::DeviceBuffer<u32> not_accreted;
        sham::DeviceBuffer<u32> accreted;
    };

    for (size_t sink_id = 0; sink_id < sink_parts.size(); sink_id++) {
        Sink &s = sink_parts[sink_id];

        Tvec r_sink    = s.pos;
        Tvec v_sink    = s.velocity;
        Tscal acc_rad2 = s.accretion_radius * s.accretion_radius;

        // flags particles for accretion
        shambase::DistributedData<AccretionFlagBufs> accretion_flag_bufs{};

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

            sham::DeviceBuffer<u32> not_accreted(Nobj, dev_sched);
            sham::DeviceBuffer<u32> accreted(Nobj, dev_sched);

            sham::kernel_call(
                q,
                sham::MultiRef{buf_xyz},
                sham::MultiRef{not_accreted, accreted},
                Nobj,
                [r_sink, acc_rad2](
                    u32 id_a,
                    const Tvec *__restrict xyz,
                    u32 *__restrict not_acc,
                    u32 *__restrict acc) {
                    Tvec r            = xyz[id_a] - r_sink;
                    bool not_accreted = sycl::dot(r, r) > acc_rad2;
                    not_acc[id_a]     = (not_accreted) ? 1 : 0;
                    acc[id_a]         = (!not_accreted) ? 1 : 0;
                });

            accretion_flag_bufs.add_obj(
                cur_p.id_patch, AccretionFlagBufs{std::move(not_accreted), std::move(accreted)});
        });

        // list the ids that will be accreted
        shambase::DistributedData<sham::DeviceBuffer<u32>> bufs_id_list_accrete{};

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sham::DeviceBuffer<u32> &accreted = accretion_flag_bufs.get(cur_p.id_patch).accreted;

            sham::DeviceBuffer<u32> id_list_accrete
                = shamalgs::stream_compact(dev_sched, accreted, Nobj);

            bufs_id_list_accrete.add_obj(cur_p.id_patch, std::move(id_list_accrete));
        });

        // compute the accreted mass, position moment and linear momentum
        Tscal s_acc_mass = 0;
        Tvec s_acc_mxyz  = {0, 0, 0};
        Tvec s_acc_pxyz  = {0, 0, 0};
        Tvec s_acc_maxyz = {0, 0, 0};
        Tvec s_acc_lxyz  = {0, 0, 0};

        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::DeviceBuffer<Tvec> &buf_axyz = pdat.get_field_buf_ref<Tvec>(iaxyz);

            sham::DeviceBuffer<u32> &id_list_accrete = bufs_id_list_accrete.get(cur_p.id_patch);

            // sum accreted values onto sink
            if (id_list_accrete.get_size() > 0) {
                u32 Naccrete = shambase::narrow_or_throw<u32>(id_list_accrete.get_size());

                Tscal acc_mass = gpart_mass * Naccrete;

                sham::DeviceBuffer<Tvec> pxyz_acc(Naccrete, dev_sched);
                sham::DeviceBuffer<Tvec> maxyz_acc(Naccrete, dev_sched);
                sham::DeviceBuffer<Tvec> mxyz_acc(Naccrete, dev_sched);
                sham::DeviceBuffer<Tvec> lxyz_acc(Naccrete, dev_sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz, buf_vxyz, buf_axyz, id_list_accrete},
                    sham::MultiRef{pxyz_acc, mxyz_acc, maxyz_acc, lxyz_acc},
                    Naccrete,
                    [gpart_mass, r_sink, v_sink, dt](
                        u32 id_a,
                        const Tvec *__restrict xyz,
                        const Tvec *__restrict vxyz,
                        const Tvec *__restrict axyz,
                        const u32 *__restrict id_acc,
                        Tvec *__restrict accretion_p,
                        Tvec *__restrict accretion_mr,
                        Tvec *__restrict accretion_ma,
                        Tvec *__restrict accretion_l) {
                        u32 i_a            = id_acc[id_a];
                        Tvec r             = xyz[i_a];
                        Tvec v             = vxyz[i_a];
                        Tvec a             = axyz[i_a];
                        accretion_p[id_a]  = gpart_mass * v;
                        accretion_mr[id_a] = gpart_mass * r;
                        accretion_ma[id_a] = gpart_mass * a;

                        // dirty trick to account for the residual acceleration in the spin. This
                        // allows us to maitain a much better angular momentum conservation.
                        v += a * dt / 2;
                        accretion_l[id_a] = gpart_mass * sycl::cross(r - r_sink, v - v_sink);
                    });

                Tvec acc_pxyz  = shamalgs::primitives::sum(dev_sched, pxyz_acc, 0, Naccrete);
                Tvec acc_mxyz  = shamalgs::primitives::sum(dev_sched, mxyz_acc, 0, Naccrete);
                Tvec acc_maxyz = shamalgs::primitives::sum(dev_sched, maxyz_acc, 0, Naccrete);
                Tvec acc_lxyz  = shamalgs::primitives::sum(dev_sched, lxyz_acc, 0, Naccrete);

                s_acc_mass += acc_mass;
                s_acc_pxyz += acc_pxyz;
                s_acc_mxyz += acc_mxyz;
                s_acc_maxyz += acc_maxyz;
                s_acc_lxyz += acc_lxyz;
            }
        });

        Tscal sum_acc_mass = shamalgs::collective::allreduce_sum(s_acc_mass);

        // if there is accretion continue otherwise skip that part
        if (sum_acc_mass <= 0) {
            continue;
        }

        Tvec sum_acc_pxyz  = shamalgs::collective::allreduce_sum(s_acc_pxyz);
        Tvec sum_acc_mxyz  = shamalgs::collective::allreduce_sum(s_acc_mxyz);
        Tvec sum_acc_maxyz = shamalgs::collective::allreduce_sum(s_acc_maxyz);
        Tvec sum_acc_lxyz  = shamalgs::collective::allreduce_sum(s_acc_lxyz);

        // compute the new sink values
        Tscal new_mass   = s.mass + sum_acc_mass;
        Tvec new_pos     = (sum_acc_mxyz + s.pos * s.mass) / (s.mass + sum_acc_mass);
        Tvec new_vel     = (sum_acc_pxyz + s.velocity * s.mass) / (s.mass + sum_acc_mass);
        Tvec new_acc     = (sum_acc_maxyz + s.sph_acceleration * s.mass) / (s.mass + sum_acc_mass);
        Tvec new_ang_mom = s.angular_momentum + sum_acc_lxyz
                           - new_mass * sycl::cross(new_pos - s.pos, new_vel - s.velocity);

        // write back the updated sink state
        auto new_state             = s;
        new_state.mass             = new_mass;
        new_state.pos              = new_pos;
        new_state.velocity         = new_vel;
        new_state.angular_momentum = new_ang_mom;
        new_state.sph_acceleration = new_acc;

        had_accretion = true;
        log += shambase::format(
            "\n    id {} deltas : mass={} r={} v={} l={}",
            sink_id,
            new_state.mass - s.mass,
            new_state.pos - s.pos,
            new_state.velocity - s.velocity,
            new_state.angular_momentum - s.angular_momentum);

        s = new_state;

        // evict accreted particles from patches
        scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
            u32 Nobj = pdat.get_obj_cnt();

            sham::DeviceBuffer<u32> &not_accreted
                = accretion_flag_bufs.get(cur_p.id_patch).not_accreted;
            sham::DeviceBuffer<u32> &accreted = accretion_flag_bufs.get(cur_p.id_patch).accreted;

            sham::DeviceBuffer<u32> &id_list_accrete = bufs_id_list_accrete.get(cur_p.id_patch);

            if (id_list_accrete.get_size() > 0) {

                sham::DeviceBuffer<u32> id_list_keep
                    = shamalgs::stream_compact(dev_sched, not_accreted, Nobj);

                pdat.keep_ids(
                    id_list_keep, shambase::narrow_or_throw<u32>(id_list_keep.get_size()));
            }
        });
    }

    if (shamcomm::world_rank() == 0 && had_accretion) {
        logger::info_ln("sph::Sink", log);
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::predictor_step(Tscal dt) {

    StackEntry stack_loc{};

    if (storage.sinks.is_empty()) {
        return;
    }

    compute_ext_forces(dt);

    std::vector<Sink> &sink_parts = storage.sinks.get();

    for (Sink &s : sink_parts) {
        s.velocity += (dt / 2) * (s.sph_acceleration + s.ext_acceleration);
    }

    for (Sink &s : sink_parts) {
        s.pos += (dt) *s.velocity;
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::corrector_step(Tscal dt) {

    StackEntry stack_loc{};

    if (storage.sinks.is_empty()) {
        return;
    }

    std::vector<Sink> &sink_parts = storage.sinks.get();

    for (Sink &s : sink_parts) {
        s.velocity += (dt / 2) * (s.sph_acceleration + s.ext_acceleration);
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_sph_forces() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    if (storage.sinks.is_empty()) {
        return;
    }

    std::vector<Sink> &sink_parts = storage.sinks.get();

    Tscal G            = solver_config.get_constant_G();
    Tscal epsilon_grav = 1e-9;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 iaxyz_ext       = pdl.get_field_idx<Tvec>("axyz_ext");

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    std::vector<Tvec> result_acc_sinks{};

    for (Sink &s : sink_parts) {

        Tvec sph_acc_sink = {};

        scheduler().for_each_patchdata_nonempty(
            [&, G, epsilon_grav, gpart_mass](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(ixyz);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::DeviceBuffer<Tvec> buf_sync_axyz(pdat.get_obj_cnt(), dev_sched);

                Tscal sink_mass = s.mass;
                Tscal sink_racc = s.accretion_radius;
                Tvec sink_pos   = s.pos;

                sham::EventList depends_list;
                auto xyz       = buf_xyz.get_read_access(depends_list);
                auto axyz_ext  = buf_axyz_ext.get_write_access(depends_list);
                auto axyz_sync = buf_sync_axyz.get_write_access(depends_list);

                auto e = q.submit(
                    depends_list,
                    [&, G, epsilon_grav, sink_mass, sink_pos, sink_racc](sycl::handler &cgh) {
                        shambase::parallel_for(
                            cgh, pdat.get_obj_cnt(), "sink-sph forces", [=](i32 id_a) {
                                Tvec r_a = xyz[id_a];

                                Tvec delta = r_a - sink_pos;
                                Tscal d    = sycl::length(delta);

                                Tvec force = G * delta / (d * d * d);

                                // This is a hack to avoid the sink kaboom effect
                                // when the particle is being advected close to the sink before
                                // being accreted
                                if (d < sink_racc) {
                                    force = {0, 0, 0};
                                }

                                axyz_sync[id_a] = force * gpart_mass;
                                axyz_ext[id_a] += -force * sink_mass;
                            });
                    });

                buf_xyz.complete_event_state(e);
                buf_axyz_ext.complete_event_state(e);
                buf_sync_axyz.complete_event_state(e);

                sph_acc_sink
                    += shamalgs::primitives::sum(dev_sched, buf_sync_axyz, 0, pdat.get_obj_cnt());
            });

        result_acc_sinks.push_back(sph_acc_sink);
    }

    std::vector<Tvec> gathered_result_acc_sinks{};
    shamalgs::collective::vector_allgatherv(
        result_acc_sinks, gathered_result_acc_sinks, MPI_COMM_WORLD);

    u32 id_s = 0;
    for (Sink &s : sink_parts) {

        s.sph_acceleration = {};

        for (u32 rid = 0; rid < shamcomm::world_size(); rid++) {
            s.sph_acceleration += gathered_result_acc_sinks[rid * sink_parts.size() + id_s];
        }

        id_s++;
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_ext_forces(Tscal dt) {

    StackEntry stack_loc{};

    if (storage.sinks.is_empty()) {
        return;
    }

    std::vector<Sink> &sink_parts = storage.sinks.get();

    // In the following part of the code, we calculate the acceleration depending of the solver
    // config( Orbital precession, Spin-Orbit, Spin-Spin, Radiation Reaction) Note that all these
    // terms (except for the Newton) are only true for binary (two sinks)
    bool OP = solver_config.compute_OP;
    bool SO = solver_config.compute_SO;
    bool SS = solver_config.compute_SS;
    bool RR = solver_config.compute_RR;

    logger::info_ln("-------- SinkParticleUpdate: Post-Newtonian terms --------");
    logger::info_ln("1PN", solver_config.compute_OP);
    logger::info_ln("SO", solver_config.compute_SO);
    logger::info_ln("SS", solver_config.compute_SS);
    logger::info_ln("RR", solver_config.compute_RR);

    for (Sink &s : sink_parts) {
        s.ext_acceleration = Tvec{};
    }
    // Definition of G and c
    Tscal G = solver_config.get_constant_G();
    Tscal c = solver_config.get_constant_c();

    Tscal epsilon_grav_sink = 1e-9;

    for (Sink &s1 : sink_parts) {

        Tvec sum{};

        for (Sink &s2 : sink_parts) {

            Tscal M   = s1.mass + s2.mass;
            Tscal nu  = s1.mass * s2.mass / M;
            Tscal eta = nu / M;

            if (&s1 == &s2)
                continue;

            Tvec term0{};
            Tvec term1{};
            Tvec term2{};
            Tvec term3{};
            Tvec term4{};

            Tvec rij       = s1.pos - s2.pos;
            Tscal rij_scal = sycl::length(rij);

            Tvec nij = rij / (rij_scal+epsilon_grav_sink);
            Tvec vij = s1.velocity - s2.velocity;

            Tscal vij_nij = sycl::dot(vij, nij);
            Tscal v2      = sycl::dot(vij, vij);
            Tvec S1       = s1.angular_momentum;
            Tvec S2       = s2.angular_momentum;
            Tvec S        = S1 + S2;
            Tvec Delta    = M * (s2.angular_momentum / s2.mass - s1.angular_momentum / s1.mass);
            Tscal dm      = s1.mass - s2.mass;

            term0 = -G * M * rij / (rij_scal * rij_scal * rij_scal + epsilon_grav_sink);
            sum += s2.mass / M * term0;

            if (OP) {
                term1 = -G * M / (rij_scal * rij_scal + epsilon_grav_sink)
                        * (((1 + 3 * eta) * v2 * nij)
                           - 2.0 * (2 + eta) * G * M / ((rij_scal + epsilon_grav_sink)) * nij
                           - 1.5 * eta * vij_nij * vij_nij * nij - 2.0 * (2 - eta) * vij_nij * vij);
                sum += 1 / (c * c) * s2.mass / M * term1;
            }

            if (SO) {
                term2 = G / (c * c * (rij_scal * rij_scal * rij_scal + epsilon_grav_sink))
                        * (6 * nij * (sycl::dot(sycl::cross(nij, vij), 2 * S + dm / M * Delta))
                           - sycl::cross(vij, 7 * S + 3 * dm / M * Delta)
                           + + 3*vij_nij*sycl::cross(nij, 3*S + dm/M*Delta));


                sum += s2.mass / M * term2;
            }

            if (SS) {
                term3
                    = -3*G
                      / (c * c * nu
                         * (rij_scal * rij_scal * rij_scal * rij_scal + epsilon_grav_sink))
                      * (nij * sycl::dot(S1, S2) + S1 * sycl::dot(nij, S2) + S2 * sycl::dot(nij, S1)
                         - 5 * nij * sycl::dot(nij, S1) * sycl::dot(nij, S2));

                sum += s2.mass / M * term3;
            }
            if (RR) {
                term4 = 8.0 / 5.0 * G * G * eta * M * M
                        / (c * c * c * c * c * (rij_scal * rij_scal * rij_scal + epsilon_grav_sink))
                        * (vij_nij * nij
                               * (18 * v2 + 2.0 / 3.0 * G * M / (rij_scal + epsilon_grav_sink)
                                  - 25 * vij_nij * vij_nij)
                           - (6 * v2 - 2 * G * M / (rij_scal + epsilon_grav_sink)
                              - 15 * vij_nij * vij_nij)
                                 * vij);

                sum += s2.mass / M * term4;
            }
        }
        s1.ext_acceleration += sum;
    }

    update_sink_spins(dt);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::update_sink_spins(Tscal dt) {

    // Definition of the constants G and c for the calculations of spin precession (the same as in
    // the compute_ext_forces function)
    Tscal G = solver_config.get_constant_G(); // G=4*pi*2
    Tscal c = solver_config.get_constant_c(); // c= 63 241.077 AU/year

    if (storage.sinks.is_empty()) {
        return;
    }

    std::vector<Sink> &sink_parts = storage.sinks.get();

    Tscal epsilon_spin = 1e-9;

    for (Sink &s1 : sink_parts) {
        Tvec dS = {};

        for (Sink &s2 : sink_parts) {
            if (&s1 == &s2) {
                continue;
            }
            Tvec rij = s1.pos - s2.pos;
            Tvec vij = s1.velocity - s2.velocity;
            Tscal m1 = s1.mass;
            Tscal m2 = s2.mass;
            Tscal M  = m1 + m2;
            Tscal nu = m1 * m2 / M;
            Tvec L   = nu * sycl::cross(rij, vij);

            Tscal rij_scal  = sycl::length(rij) + epsilon_spin;
            Tvec nij        = rij / (rij_scal + epsilon_spin);
            Tvec S1         = s1.angular_momentum;
            Tvec S2         = s2.angular_momentum;
            Tscal prefactor = G / (c * c * rij_scal * rij_scal * rij_scal);
            // Simple spin precession structure.

            Tvec Omega_prec
                = prefactor * ((2 + 3.0 * m2 / (2.0 * m1)) * L + S2 - 3 * sycl::dot(nij, S2) * nij);

            dS += sycl::cross(Omega_prec, S1);
        }

        s1.angular_momentum += dS * dt;
    }
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
