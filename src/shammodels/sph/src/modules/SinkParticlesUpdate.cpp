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

#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::corrector_step(Tscal dt) {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &vel  = get_sink_vel<Tvec>(sync);
    if (vel.empty()) {
        return;
    }

    auto &acc_sph = get_sink_acc_sph<Tvec>(sync);
    auto &acc_ext = get_sink_acc_ext<Tvec>(sync);

    for (size_t i = 0; i < vel.size(); i++) {
        vel[i] += (dt / 2) * (acc_sph[i] + acc_ext[i]);
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_sph_forces() {

    StackEntry stack_loc{};

    Tscal gpart_mass = solver_config.gpart_mass;

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    auto &mass             = get_sink_mass<Tvec>(sync);
    auto &accretion_radius = get_sink_accretion_radius<Tvec>(sync);
    auto &acc_sph          = get_sink_acc_sph<Tvec>(sync);

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

    for (size_t sink_id = 0; sink_id < pos.size(); sink_id++) {

        Tvec sph_acc_sink = {};

        scheduler().for_each_patchdata_nonempty(
            [&, G, epsilon_grav, gpart_mass](Patch cur_p, PatchDataLayer &pdat) {
                sham::DeviceBuffer<Tvec> &buf_xyz      = pdat.get_field_buf_ref<Tvec>(ixyz);
                sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

                sham::DeviceBuffer<Tvec> buf_sync_axyz(pdat.get_obj_cnt(), dev_sched);

                Tscal sink_mass = mass[sink_id];
                Tscal sink_racc = accretion_radius[sink_id];
                Tvec sink_pos   = pos[sink_id];

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{buf_axyz_ext, buf_sync_axyz},
                    pdat.get_obj_cnt(),
                    [G, sink_mass, sink_pos, sink_racc, gpart_mass](
                        u32 id_a,
                        const Tvec *__restrict xyz,
                        Tvec *__restrict axyz_ext,
                        Tvec *__restrict axyz_sync) {
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

                sph_acc_sink
                    += shamalgs::primitives::sum(dev_sched, buf_sync_axyz, 0, pdat.get_obj_cnt());
            });

        result_acc_sinks.push_back(sph_acc_sink);
    }

    std::vector<Tvec> gathered_result_acc_sinks{};
    shamalgs::collective::vector_allgatherv(
        result_acc_sinks, gathered_result_acc_sinks, MPI_COMM_WORLD);

    for (size_t id_s = 0; id_s < pos.size(); id_s++) {

        acc_sph[id_s] = {};

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
            acc_sph[id_s] += gathered_result_acc_sinks[rid * pos.size() + id_s];
        }
    }
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
