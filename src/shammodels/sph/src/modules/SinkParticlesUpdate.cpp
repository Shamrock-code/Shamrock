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
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamsolvergraph/IFreeable.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <memory>
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
                                                                                                   \
    /* ------------------- (sink) inputs ------------------- */                                    \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_accr_radii)                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    /* sink_accretion_table[id_a] = who should accrete part [id_a] (or u32_max if none); */        \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)

namespace shammodels::common::modules {
    template<class Tvec>
    class SinkParticlesFlagAccreteHard : public shamrock::solvergraph::INode,
                                         public shamrock::solvergraph::IFreeable {

        using Tscal = shambase::VecComponent<Tvec>;

        std::unique_ptr<sham::DeviceBuffer<Tvec>> sink_pos;
        std::unique_ptr<sham::DeviceBuffer<Tscal>> sink_accr_radii;

        public:
        SinkParticlesFlagAccreteHard() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            auto &q        = shambase::get_check_ref(dev_sched).get_queue();

            auto &sink_positions = edges.sink_positions.data;
            auto &sink_radii     = edges.sink_accr_radii.data;

            if (sink_positions.size() != sink_radii.size()) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "Sink positions and accretion radii must have the same size");
            }

            if (!sink_pos) {
                sink_pos
                    = std::make_unique<sham::DeviceBuffer<Tvec>>(sink_positions.size(), dev_sched);
            }
            if (!sink_accr_radii) {
                sink_accr_radii
                    = std::make_unique<sham::DeviceBuffer<Tscal>>(sink_radii.size(), dev_sched);
            }

            sink_pos->resize(sink_positions.size());
            sink_accr_radii->resize(sink_radii.size());

            sink_pos->copy_from_stdvec(sink_positions);
            sink_accr_radii->copy_from_stdvec(sink_radii);

            edges.positions.check_sizes(edges.part_counts.indexes);
            edges.sink_accretion_table.ensure_sizes(edges.part_counts.indexes);

            auto &pos_spans       = edges.positions.get_spans();
            auto &table_acc_spans = edges.sink_accretion_table.get_spans();

            u32 sink_count = shambase::narrow_or_throw<u32>(sink_positions.size());

            edges.part_counts.indexes.for_each([&](u64 id_patch, u32 part_count) {
                sham::kernel_call(
                    q,
                    sham::MultiRef{pos_spans.get(id_patch), *sink_pos, *sink_accr_radii},
                    sham::MultiRef{table_acc_spans.get(id_patch)},
                    part_count,
                    [sink_count](
                        u32 id_a,
                        const Tvec *__restrict part_pos,
                        const Tvec *__restrict sink_pos,
                        const Tscal *__restrict sink_accr_radii,
                        u32 *__restrict sink_accretion_table) {
                        Tvec r_a = part_pos[id_a];

                        u32 result = u32_max;

                        for (u32 i_sink = 0; i_sink < sink_count; i_sink++) {
                            Tscal acc_radii = sink_accr_radii[i_sink];
                            Tvec d          = r_a - sink_pos[i_sink];

                            bool should_accrete = sycl::dot(d, d) <= acc_radii * acc_radii;
                            if (should_accrete) {
                                result = i_sink;
                                break;
                            }
                        }

                        sink_accretion_table[id_a] = result;
                    });
            });
        }

        inline void free_alloc() {
            sink_pos        = {};
            sink_accr_radii = {};
        }

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesFlagAccreteHard";
        }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (param) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
                                                                                                   \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, velocities)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, accelerations)                                   \
                                                                                                   \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RW(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- (sink) in/out ------------------- */                                    \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_velocities)                     \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_accelerations)                  \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_angmom)                         \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_mass)

namespace shammodels::common::modules {
    template<class Tvec>
    class SinkParticlesAccreteQuantities : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkParticlesAccreteQuantities() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

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
                        [i_sink](
                            u32 id_a, const u32 *__restrict acc_table, u32 *__restrict acc_flag) {
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
                                accretion_l[id_a]
                                    = gpart_mass * sycl::cross(r - r_sink, v - v_sink);
                            });

                        Tvec acc_pxyz = shamalgs::primitives::sum(dev_sched, pxyz_acc, 0, Naccrete);
                        Tvec acc_mxyz = shamalgs::primitives::sum(dev_sched, mxyz_acc, 0, Naccrete);
                        Tvec acc_maxyz
                            = shamalgs::primitives::sum(dev_sched, maxyz_acc, 0, Naccrete);
                        Tvec acc_lxyz = shamalgs::primitives::sum(dev_sched, lxyz_acc, 0, Naccrete);

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
                log += shambase::format(
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

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesAccreteQuantities";
        }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (sink) accretion table ------------------- */                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::Field<u32>, sink_accretion_table)                                  \
                                                                                                   \
    /* ------------------- Patchdatas ------------------- */                                       \
    X_RW(shamrock::solvergraph::PatchDataLayerRefs, pdats)

namespace shammodels::common::modules {
    template<class Tvec>
    class SinkParticlesEvictAccretedParticles : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkParticlesEvictAccretedParticles() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            auto &q        = shambase::get_check_ref(dev_sched).get_queue();

            sham::DeviceBuffer<u32> keep_flag(0, dev_sched);
            sham::DeviceBuffer<int> accr_flag(1, dev_sched);

            edges.part_counts.indexes.for_each([&](u64 id_patch, u32 Nobj) {
                auto &pdat      = edges.pdats.get(id_patch);
                auto &acc_table = edges.sink_accretion_table.get_spans().get(id_patch);

                keep_flag.resize(Nobj);
                accr_flag.fill(0);

                sham::kernel_call(
                    q,
                    sham::MultiRef{acc_table},
                    sham::MultiRef{keep_flag, accr_flag},
                    Nobj,
                    [](u32 id_a,
                       const u32 *__restrict acc_table,
                       u32 *__restrict keep_flag,
                       int *__restrict accr_flag) {
                        bool keep       = acc_table[id_a] == u32_max;
                        keep_flag[id_a] = keep ? 1 : 0;

                        sycl::atomic_ref<
                            int,
                            sycl::memory_order_relaxed,
                            sycl::memory_scope_device,
                            sycl::access::address_space::global_space>
                            atomic_accr(accr_flag[0]);

                        if (!keep) {
                            atomic_accr.fetch_or(1);
                        }
                    });

                int accr_flag_val = accr_flag.get_val_at_idx(0);

                if (accr_flag_val != 0) {

                    sham::DeviceBuffer<u32> id_list_keep
                        = shamalgs::stream_compact(dev_sched, keep_flag, Nobj);

                    pdat.keep_ids(
                        id_list_keep, shambase::narrow_or_throw<u32>(id_list_keep.get_size()));
                }
            });
        }

        inline virtual std::string _impl_get_label() const {
            return "SinkParticlesEvictAccretedParticles";
        }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules
#undef NODE_EDGES

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

    shammodels::common::modules::SinkParticlesFlagAccreteHard<Tvec> flag_node;
    flag_node.set_edges(
        part_counts, positions, sink_positions, sink_accr_radii, sink_accretion_table);
    flag_node.evaluate();

    shammodels::common::modules::SinkParticlesAccreteQuantities<Tvec> qty_node;
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

    shammodels::common::modules::SinkParticlesEvictAccretedParticles<Tvec> evict_node;
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
