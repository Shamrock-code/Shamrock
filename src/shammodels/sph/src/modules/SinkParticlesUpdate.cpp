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
#include "shamalgs/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shamalgs/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/matrix.hpp"
#include "shammath/matrix_op.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shamsys/legacy/log.hpp"
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

    /* // Uncomment this if you want to compute total momentum to check some stuff
    auto get_tot_mom = [&]() -> Tvec {
        Tvec total_momentum = {};

        sham::DeviceBuffer<Tvec> total_momentum_part(0, dev_sched);

        scheduler().for_each_patchdata_nonempty(
            [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer
    &pdat) { u32 len = pdat.get_obj_cnt();

                total_momentum_part.resize(len);

                sham::DeviceBuffer<Tvec> &vxyz_buf =
    pdat.get_field_buf_ref<Tvec>(ivxyz);

                sham::kernel_call(
                    q,
                    sham::MultiRef{vxyz_buf},
                    sham::MultiRef{total_momentum_part},
                    len,
                    [gpart_mass](
                        u32 i, const Tvec *__restrict vxyz, Tvec *__restrict
    total_momentum_part) { total_momentum_part[i] = gpart_mass * vxyz[i];
                    });

                total_momentum += shamalgs::primitives::sum(dev_sched,
    total_momentum_part, 0, len);
            });

        Tvec tot_total_momentum =
    shamalgs::collective::allreduce_sum(total_momentum);

        for (auto &sink : sink_parts) {
            tot_total_momentum += sink.mass * sink.velocity;
        }

        return tot_total_momentum;
    };
  */

    auto get_ang_mom = [&]() -> Tvec {
        Tvec angular_momentum = {};

        sham::DeviceBuffer<Tvec> total_momentum_part(0, dev_sched);

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            u32 len = pdat.get_obj_cnt();

            total_momentum_part.resize(len);

            sham::DeviceBuffer<Tvec> &xyz_buf  = pdat.get_field_buf_ref<Tvec>(ixyz);
            sham::DeviceBuffer<Tvec> &vxyz_buf = pdat.get_field_buf_ref<Tvec>(ivxyz);
            sham::kernel_call(
                q,
                sham::MultiRef{xyz_buf, vxyz_buf},
                sham::MultiRef{total_momentum_part},
                len,
                [gpart_mass](
                    u32 i,
                    const Tvec *__restrict xyz,
                    const Tvec *__restrict vxyz,
                    Tvec *__restrict total_momentum_part) {
                    total_momentum_part[i] = gpart_mass * sycl::cross(xyz[i], vxyz[i]);
                });

            angular_momentum += shamalgs::primitives::sum(dev_sched, total_momentum_part, 0, len);
        });

        Tvec tot_angular_momentum = shamalgs::collective::allreduce_sum(angular_momentum);

        for (auto &sink : sink_parts) {
            tot_angular_momentum
                += sink.mass * sycl::cross(sink.pos, sink.velocity) + sink.angular_momentum;
        }
        return tot_angular_momentum;
    };

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

                        // dirty trick to account for the residual acceleration in the
                        // spin. This allows us to maitain a much better angular momentum
                        // conservation.
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

    // now time for torque free torque restitution
    for (Sink &s : sink_parts) {
        if (s.is_torque_free && sham::dot(s.angular_momentum, s.angular_momentum) > 0) {
            // boost particles in a radius of 4 racc to empty the sink angular_momentum

            struct BoostWeight {
                Tscal radius;
                Tscal weight(Tscal r) const {
                    if (r < radius * Kernel::Rkern) {
                        return Kernel::W_3d(r, radius);
                    } else {
                        return 0;
                    }
                }
            };

            BoostWeight weight_func{
                s.accretion_radius * s.torque_boost_radius_fact / Kernel::Rkern};

            Tvec r_sink = s.pos;

            using mat = shammath::mat<Tscal, 3, 3>;

            mat I_rank{};

            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                u32 Nobj = pdat.get_obj_cnt();

                sham::DeviceBuffer<Tvec> &buf_xyz = pdat.get_field_buf_ref<Tvec>(ixyz);

                sham::DeviceBuffer<f64> weighted_intertia(9 * Nobj, dev_sched);
                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{weighted_intertia},
                    Nobj,
                    [r_sink, gpart_mass, Nobj, weight_func](
                        u32 id_a, const Tvec *__restrict xyz, Tscal *weighted_intertia) {
                        Tvec r_a   = xyz[id_a] - r_sink;
                        Tscal r_a2 = sham::dot(r_a, r_a);

                        Tscal w = gpart_mass * weight_func.weight(sycl::sqrt(r_a2));

                        if (w == 0) {
                            return;
                        }

                        mat I{};

                        { // I = w [Id r_a^2 - r_a r_a^T]
                            auto Imd = I.get_mdspan();

                            Imd(0, 0) += w * (r_a.y() * r_a.y() + r_a.z() * r_a.z());
                            Imd(1, 1) += w * (r_a.x() * r_a.x() + r_a.z() * r_a.z());
                            Imd(2, 2) += w * (r_a.x() * r_a.x() + r_a.y() * r_a.y());

                            Imd(0, 1) -= w * r_a.x() * r_a.y();
                            Imd(0, 2) -= w * r_a.x() * r_a.z();
                            Imd(1, 2) -= w * r_a.y() * r_a.z();

                            // symmetry time !
                            Imd(1, 0) = Imd(0, 1);
                            Imd(2, 0) = Imd(0, 2);
                            Imd(2, 1) = Imd(1, 2);
                        }

#pragma unroll
                        for (u32 i = 0; i < 9; i++) {
                            weighted_intertia[Nobj * i + id_a] = I.data[i];
                        }
                    });

                for (u32 i = 0; i < 9; i++) {
                    I_rank.data[i] += shamalgs::primitives::sum(
                        dev_sched, weighted_intertia, Nobj * i, Nobj * (i + 1));
                }
            });

            mat weighted_intertia_sum{};

            for (u32 i = 0; i < 9; i++) {
                weighted_intertia_sum.data[i] = shamalgs::collective::allreduce_sum(I_rank.data[i]);
            }

            mat I_inv{};
            if (shammath::mat_det_33(weighted_intertia_sum.get_mdspan()) != 0) {
                shammath::mat_inv_33(weighted_intertia_sum.get_mdspan(), I_inv.get_mdspan());
            }

            shammath::vec<Tscal, 3> delta_l{};
            delta_l.data[0] = s.angular_momentum.x();
            delta_l.data[1] = s.angular_momentum.y();
            delta_l.data[2] = s.angular_momentum.z();

            shammath::vec<Tscal, 3> delta_w{};
            shammath::mat_prod(
                I_inv.get_mdspan(), delta_l.get_mdspan_mat_col(), delta_w.get_mdspan_mat_col());

            Tvec vec_delta_w = {delta_w.data[0], delta_w.data[1], delta_w[2]};

            logger::raw_ln("delta l =", delta_l.data[0], delta_l.data[1], delta_l.data[2]);
            logger::raw_ln("delta w =", delta_w.data[0], delta_w.data[1], delta_w.data[2]);

            Tvec lboost_sum_rank{};
            Tvec pboost_sum_rank{};
            scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                u32 Nobj = pdat.get_obj_cnt();

                sham::DeviceBuffer<Tvec> &buf_xyz  = pdat.get_field_buf_ref<Tvec>(ixyz);
                sham::DeviceBuffer<Tvec> &buf_vxyz = pdat.get_field_buf_ref<Tvec>(ivxyz);

                sham::DeviceBuffer<Tvec> l_boost(Nobj, dev_sched);
                sham::DeviceBuffer<Tvec> p_boost(Nobj, dev_sched);

                sham::kernel_call(
                    q,
                    sham::MultiRef{buf_xyz},
                    sham::MultiRef{buf_vxyz, l_boost, p_boost},
                    Nobj,
                    [r_sink, delta_w = vec_delta_w, weight_func, gpart_mass](
                        u32 id_a,
                        const Tvec *__restrict xyz,
                        Tvec *__restrict vxyz,
                        Tvec *__restrict lboost,
                        Tvec *__restrict pboost) {
                        Tvec r_a   = xyz[id_a] - r_sink;
                        Tscal r_a2 = sham::dot(r_a, r_a);

                        Tscal W = weight_func.weight(sycl::sqrt(r_a2));

                        Tvec v_boost = W * sycl::cross(delta_w, r_a);

                        vxyz[id_a] += v_boost;
                        lboost[id_a] += gpart_mass * sycl::cross(r_a, v_boost);
                        pboost[id_a] += gpart_mass * v_boost;
                    });

                lboost_sum_rank += shamalgs::primitives::sum(dev_sched, l_boost, 0, Nobj);
                pboost_sum_rank += shamalgs::primitives::sum(dev_sched, p_boost, 0, Nobj);
            });

            Tvec lboost = shamalgs::collective::allreduce_sum(lboost_sum_rank);
            Tvec pboost = shamalgs::collective::allreduce_sum(pboost_sum_rank);

            logger::raw_ln("lboost =", lboost, "pboost =", pboost);

            s.angular_momentum -= lboost;
            s.velocity -= pboost / s.mass;
        }
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

    compute_ext_forces();

    std::vector<Sink> &sink_parts = storage.sinks.get();

    for (Sink &s : sink_parts) {
        s.velocity += (dt / 2) * s.sph_acceleration;
    }

    for (Sink &s : sink_parts) {
        s.velocity += (dt / 2) * s.ext_acceleration;
    }

    for (Sink &s : sink_parts) {
        s.pos += (dt) *s.velocity;
    }

    for (Sink &s : sink_parts) {
        s.velocity += (dt / 2) * s.ext_acceleration;
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
        s.velocity += (dt / 2) * s.sph_acceleration;
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

                                // This is a hack to avoid the sink
                                // kaboom effect when the particle is
                                // being advected close to the sink
                                // before being accreted
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
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_ext_forces() {

    StackEntry stack_loc{};

    if (storage.sinks.is_empty()) {
        return;
    }

    std::vector<Sink> &sink_parts = storage.sinks.get();

    for (Sink &s : sink_parts) {
        s.ext_acceleration = Tvec{};
    }

    Tscal G                 = solver_config.get_constant_G();
    Tscal epsilon_grav_sink = 1e-9;

    for (Sink &s1 : sink_parts) {
        Tvec sum{};
        for (Sink &s2 : sink_parts) {
            Tvec rij       = s1.pos - s2.pos;
            Tscal rij_scal = sycl::length(rij);
            sum -= G * s2.mass * rij / (rij_scal * rij_scal * rij_scal + epsilon_grav_sink);
        }
        s1.ext_acceleration = sum;
    }
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
