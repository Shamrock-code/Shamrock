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
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesAccreteQuantities.hpp"
#include "shammodels/sph/modules/SinkParticlesEvictAccretedParticles.hpp"
#include "shammodels/sph/modules/SinkParticlesFlagAccreteHard.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include <memory>
#include <vector>

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::accrete_particles(Tscal dt) {
    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    if (!has_sinks<Tvec>(sync)) {
        return;
    }

    using namespace shamrock;
    using namespace shamrock::patch;
    using namespace shamrock::solvergraph;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();
    const u32 ixyz            = pdl.get_field_idx<Tvec>("xyz");
    const u32 ivxyz           = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz           = pdl.get_field_idx<Tvec>("axyz");

    auto part_counts   = Indexes<u32>::make_shared("part_counts", "N");
    auto positions     = std::make_shared<FieldRefs<Tvec>>("xyz", "\\mathbf{r}");
    auto velocities    = std::make_shared<FieldRefs<Tvec>>("vxyz", "\\mathbf{v}");
    auto accelerations = std::make_shared<FieldRefs<Tvec>>("axyz", "\\mathbf{a}");
    auto sink_accretion_table
        = std::make_shared<Field<u32>>(1, "sink_accretion_table", "\\mathrm{acc}");
    auto pdats = std::make_shared<PatchDataLayerRefs>("patchdatas", "\\mathbb{U}");

    DDPatchDataFieldRef<Tvec> pos_dd;
    DDPatchDataFieldRef<Tvec> vel_dd;
    DDPatchDataFieldRef<Tvec> acc_dd;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u64 id = cur_p.id_patch;
        part_counts->indexes.add_obj(id, pdat.get_obj_cnt());
        pos_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(ixyz)));
        vel_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(ivxyz)));
        acc_dd.add_obj(id, std::ref(pdat.get_field<Tvec>(iaxyz)));
        pdats->patchdatas.add_obj(id, std::ref(pdat));
    });

    positions->set_refs(pos_dd);
    velocities->set_refs(vel_dd);
    accelerations->set_refs(acc_dd);

    auto gpart_mass  = IDataEdge<Tscal>::make_shared("gpart_mass", "m");
    gpart_mass->data = solver_config.gpart_mass;

    auto dt_edge  = IDataEdge<Tscal>::make_shared("dt", "dt");
    dt_edge->data = dt;

    auto sink_positions
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>("sink_pos");
    auto sink_velocities
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>("sink_vel");
    auto sink_accelerations
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>("sink_acc_sph");
    auto sink_angmom = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>(
        "sink_angular_momentum");
    auto sink_mass
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tscal>>>("sink_mass");
    auto sink_accr_radii = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tscal>>>(
        "sink_accretion_radius");

    shammodels::sph::modules::SinkParticlesFlagAccreteHard<Tvec> flag_node;
    flag_node.set_edges(
        part_counts, positions, sink_positions, sink_accr_radii, sink_accretion_table);
    flag_node.evaluate();

    shammodels::sph::modules::SinkParticlesAccreteQuantities<Tvec> qty_node;
    qty_node.set_edges(
        gpart_mass,
        dt_edge,
        part_counts,
        positions,
        velocities,
        accelerations,
        sink_accretion_table,
        sink_positions,
        sink_velocities,
        sink_accelerations,
        sink_angmom,
        sink_mass);
    qty_node.evaluate();

    shammodels::sph::modules::SinkParticlesEvictAccretedParticles<Tvec> evict_node;
    evict_node.set_edges(part_counts, sink_accretion_table, pdats);
    evict_node.evaluate();

    flag_node.free_alloc();
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::predictor_step(Tscal dt) {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    auto &vel     = get_sink_vel<Tvec>(sync);
    auto &acc_sph = get_sink_acc_sph<Tvec>(sync);
    auto &acc_ext = get_sink_acc_ext<Tvec>(sync);

    compute_ext_forces();

    for (size_t i = 0; i < pos.size(); i++) {
        vel[i] += (dt / 2) * (acc_sph[i] + acc_ext[i]);
    }

    for (size_t i = 0; i < pos.size(); i++) {
        pos[i] += dt * vel[i];
    }
}

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

    for (size_t id_s = 0; id_s < pos.size(); id_s++) {

        acc_sph[id_s] = {};

        for (u32 rid = 0; rid < shamcomm::world_size(); rid++) {
            acc_sph[id_s] += gathered_result_acc_sinks[rid * pos.size() + id_s];
        }
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_ext_forces() {

    StackEntry stack_loc{};

    auto &sync = scheduler().synchronized_data;
    auto &pos  = get_sink_pos<Tvec>(sync);
    if (pos.empty()) {
        return;
    }

    auto &mass    = get_sink_mass<Tvec>(sync);
    auto &acc_ext = get_sink_acc_ext<Tvec>(sync);

    for (size_t i = 0; i < pos.size(); i++) {
        acc_ext[i] = Tvec{};
    }

    Tscal G                 = solver_config.get_constant_G();
    Tscal epsilon_grav_sink = 1e-9;

    for (size_t i = 0; i < pos.size(); i++) {
        Tvec sum{};
        for (size_t j = 0; j < pos.size(); j++) {
            Tvec rij       = pos[i] - pos[j];
            Tscal rij_scal = sycl::length(rij);
            sum -= G * mass[j] * rij / (rij_scal * rij_scal * rij_scal + epsilon_grav_sink);
        }
        acc_ext[i] = sum;
    }
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
