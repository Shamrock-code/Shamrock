// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SinkParticlesAccreteQuantities.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/narrowing.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/modules/SinkParticlesAccreteQuantities.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <string>

namespace shammodels::sph::modules {

    template<class Tvec>
    void SinkParticlesAccreteQuantities<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        Tscal gpart_mass = edges.gpart_mass.data;
        Tscal dt         = edges.dt.data;

        sham::DeviceBuffer<u32> acc_flag(0, dev_sched);

        bool had_accretion = false;
        std::string log    = "sink accretion :";

        auto &sink_positions     = edges.sink_positions.data;
        auto &sink_velocities    = edges.sink_velocities.data;
        auto &sink_accelerations = edges.sink_accelerations.data;
        auto &sink_angmom        = edges.sink_angmom.data;
        auto &sink_mass          = edges.sink_mass.data;

        u32 sink_count = shambase::narrow_or_throw<u32>(sink_positions.size());
        for (u32 i_sink = 0; i_sink < sink_count; i_sink++) {

            Tvec r_sink = sink_positions[i_sink];
            Tvec v_sink = sink_velocities[i_sink];

            // compute the accreted mass, position moment and linear momentum
            Tscal s_acc_mass = 0;
            Tvec s_acc_mxyz  = {0, 0, 0};
            Tvec s_acc_pxyz  = {0, 0, 0};
            Tvec s_acc_maxyz = {0, 0, 0};
            Tvec s_acc_lxyz  = {0, 0, 0};

            edges.part_counts.indexes.for_each([&](u64 id_patch, u32 Nobj) {
                acc_flag.resize(Nobj);

                auto &acc_table = edges.sink_accretion_table.get_spans().get(id_patch);

                sham::kernel_call(
                    q,
                    sham::MultiRef{acc_table},
                    sham::MultiRef{acc_flag},
                    Nobj,
                    [i_sink](u32 id_a, const u32 *__restrict acc_table, u32 *__restrict acc_flag) {
                        acc_flag[id_a] = (acc_table[id_a] == i_sink) ? 1 : 0;
                    });

                auto id_list_accrete = shamalgs::stream_compact(dev_sched, acc_flag, Nobj);

                auto &pos_data = edges.positions.get_spans().get(id_patch);
                auto &vel_data = edges.velocities.get_spans().get(id_patch);
                auto &acc_data = edges.accelerations.get_spans().get(id_patch);

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
                        sham::MultiRef{pos_data, vel_data, acc_data, id_list_accrete},
                        sham::MultiRef{pxyz_acc, mxyz_acc, maxyz_acc, lxyz_acc},
                        Naccrete,
                        [r_sink, v_sink, gpart_mass, dt](
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

                            // dirty trick to account for the residual acceleration in the spin.
                            // This allows us to maitain a much better angular momentum
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

            Tscal old_mass = sink_mass[i_sink];
            Tvec old_pos   = sink_positions[i_sink];
            Tvec old_vel   = sink_velocities[i_sink];
            Tvec old_acc   = sink_accelerations[i_sink];
            Tvec old_ang   = sink_angmom[i_sink];

            // compute the new sink values
            Tscal new_mass   = old_mass + sum_acc_mass;
            Tvec new_pos     = (sum_acc_mxyz + old_pos * old_mass) / (old_mass + sum_acc_mass);
            Tvec new_vel     = (sum_acc_pxyz + old_vel * old_mass) / (old_mass + sum_acc_mass);
            Tvec new_acc     = (sum_acc_maxyz + old_acc * old_mass) / (old_mass + sum_acc_mass);
            Tvec new_ang_mom = old_ang + sum_acc_lxyz
                               - new_mass * sycl::cross(new_pos - old_pos, new_vel - old_vel);

            // write back the update sink state
            sink_mass[i_sink]          = new_mass;
            sink_positions[i_sink]     = new_pos;
            sink_velocities[i_sink]    = new_vel;
            sink_angmom[i_sink]        = new_ang_mom;
            sink_accelerations[i_sink] = new_acc;

            had_accretion = true;
            log += sham::format(
                "\n    id {} deltas : mass={} r={} v={} l={}",
                i_sink,
                new_mass - old_mass,
                new_pos - old_pos,
                new_vel - old_vel,
                new_ang_mom - old_ang);
        }

        if (shamcomm::world_rank() == 0 && had_accretion) {
            logger::info_ln("sph::Sink", log);
        }
    }

    template<class Tvec>
    std::string SinkParticlesAccreteQuantities<Tvec>::_impl_get_tex() const {
        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SinkParticlesAccreteQuantities<f64_3>;
