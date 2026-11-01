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

#include "shambase/exception.hpp"
#include "shamalgs/collective/exchanges.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shammodels/sph/sink_edges_helper.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/edge/IDataEdgeSerializable.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- (param) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, constant_G)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gpart_mass)                                      \
                                                                                                   \
    /* ------------------- (field) inputs ------------------- */                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, positions)                                       \
                                                                                                   \
    /* ------------------- (sink) inputs ------------------- */                                    \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_positions)                      \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_mass)                          \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, sink_accr_radii)                    \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, accel_ext)                                       \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, sink_acc_sph)

namespace {

    /**
     * @brief Add the gravitational interaction between the sinks and the SPH particles.
     *
     * The force exerted by the sinks is added onto the particles external acceleration, and the
     * force exerted by the SPH particles onto each sink is reduced (over all ranks) into the
     * sink SPH acceleration.
     */
    template<class Tvec>
    class SinkParticlesAddSPHForces : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkParticlesAddSPHForces() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SinkParticlesAddSPHForces"; }

        virtual std::string _impl_get_tex() const;
    };

    template<class Tvec>
    void SinkParticlesAddSPHForces<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
        sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

        Tscal G          = edges.constant_G.data;
        Tscal gpart_mass = edges.gpart_mass.data;

        const std::vector<Tvec> &sink_positions   = edges.sink_positions.data;
        const std::vector<Tscal> &sink_mass       = edges.sink_mass.data;
        const std::vector<Tscal> &sink_accr_radii = edges.sink_accr_radii.data;
        std::vector<Tvec> &sink_acc_sph           = edges.sink_acc_sph.data;

        size_t sink_count = sink_positions.size();

        if (sink_mass.size() != sink_count || sink_accr_radii.size() != sink_count
            || sink_acc_sph.size() != sink_count) {
            throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
                "sink edges size mismatch: pos={}, mass={}, accretion_radius={}, acc_sph={}",
                sink_count,
                sink_mass.size(),
                sink_accr_radii.size(),
                sink_acc_sph.size()));
        }

        edges.positions.check_sizes(edges.part_counts.indexes);
        edges.accel_ext.check_sizes(edges.part_counts.indexes);

        auto &pos_spans       = edges.positions.get_spans();
        auto &accel_ext_spans = edges.accel_ext.get_spans();

        sham::DeviceBuffer<Tvec> buf_sync_axyz(0, dev_sched);

        std::vector<Tvec> result_acc_sinks{};

        for (size_t sink_id = 0; sink_id < sink_count; sink_id++) {

            Tvec sph_acc_sink = {};

            Tscal s_mass = sink_mass[sink_id];
            Tscal s_racc = sink_accr_radii[sink_id];
            Tvec s_pos   = sink_positions[sink_id];

            edges.part_counts.indexes.for_each([&](u64 id_patch, u32 part_count) {
                buf_sync_axyz.resize(part_count);

                sham::kernel_call(
                    q,
                    sham::MultiRef{pos_spans.get(id_patch)},
                    sham::MultiRef{accel_ext_spans.get(id_patch), buf_sync_axyz},
                    part_count,
                    [s_pos, G, s_mass, s_racc, gpart_mass](
                        u32 id_a,
                        const Tvec *__restrict xyz,
                        Tvec *__restrict axyz_ext,
                        Tvec *__restrict axyz_sync) {
                        Tvec r_a = xyz[id_a];

                        Tvec delta = r_a - s_pos;
                        Tscal d    = sycl::length(delta);

                        Tvec force = G * delta / (d * d * d);

                        // This is a hack to avoid the sink kaboom effect
                        // when the particle is being advected close to the sink before
                        // being accreted
                        if (d < s_racc) {
                            force = {0, 0, 0};
                        }

                        axyz_sync[id_a] = force * gpart_mass;
                        axyz_ext[id_a] += -force * s_mass;
                    });

                sph_acc_sink += shamalgs::primitives::sum(dev_sched, buf_sync_axyz, 0, part_count);
            });

            result_acc_sinks.push_back(sph_acc_sink);
        }

        std::vector<Tvec> gathered_result_acc_sinks{};
        shamalgs::collective::vector_allgatherv(
            result_acc_sinks, gathered_result_acc_sinks, MPI_COMM_WORLD);

        for (size_t id_s = 0; id_s < sink_count; id_s++) {

            sink_acc_sph[id_s] = {};

            for (u32 rid = 0; rid < shamcomm::world_size(); rid++) {
                sink_acc_sph[id_s] += gathered_result_acc_sinks[rid * sink_count + id_s];
            }
        }
    }

    template<class Tvec>
    std::string SinkParticlesAddSPHForces<Tvec>::_impl_get_tex() const {

        auto constant_G      = get_ro_edge_base(0).get_tex_symbol();
        auto gpart_mass      = get_ro_edge_base(1).get_tex_symbol();
        auto positions       = get_ro_edge_base(3).get_tex_symbol();
        auto sink_positions  = get_ro_edge_base(4).get_tex_symbol();
        auto sink_mass       = get_ro_edge_base(5).get_tex_symbol();
        auto sink_accr_radii = get_ro_edge_base(6).get_tex_symbol();
        auto axyz_ext        = get_rw_edge_base(0).get_tex_symbol();
        auto sink_acc_sph    = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
                 Add sink / SPH particles gravitational interaction

                 \begin{align}
                 {\bf f}_{a,s} &= {constant_G} \frac{{positions}_a - {sink_positions}_s}{\vert {positions}_a - {sink_positions}_s \vert^3}
                    \quad (0 \text{ if } \vert {positions}_a - {sink_positions}_s \vert < {sink_accr_radii}_s) \\
                 {axyz_ext}_a &\mathrel{+}= - \sum_s {sink_mass}_s {\bf f}_{a,s} \\
                 {sink_acc_sph}_s &= \sum_a {gpart_mass} {\bf f}_{a,s}
                 \end{align}
             )tex";

        shambase::replace_all(tex, "{constant_G}", constant_G);
        shambase::replace_all(tex, "{gpart_mass}", gpart_mass);
        shambase::replace_all(tex, "{positions}", positions);
        shambase::replace_all(tex, "{sink_positions}", sink_positions);
        shambase::replace_all(tex, "{sink_mass}", sink_mass);
        shambase::replace_all(tex, "{sink_accr_radii}", sink_accr_radii);
        shambase::replace_all(tex, "{axyz_ext}", axyz_ext);
        shambase::replace_all(tex, "{sink_acc_sph}", sink_acc_sph);

        return tex;
    }

} // namespace

#undef NODE_EDGES

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::SinkParticlesUpdate<Tvec, SPHKernel>::compute_sph_forces() {

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
    const u32 iaxyz_ext       = pdl.get_field_idx<Tvec>("axyz_ext");

    // map the patchdata fields onto field refs
    auto part_counts   = Indexes<u32>::make_shared("part_counts", "N_{\\rm part}");
    auto xyz_refs      = FieldRefs<Tvec>::make_shared("xyz", "\\mathbf{r}");
    auto axyz_ext_refs = FieldRefs<Tvec>::make_shared("axyz_ext", "\\mathbf{a}_{\\rm ext}");

    DDPatchDataFieldRef<Tvec> xyz_field_refs      = {};
    DDPatchDataFieldRef<Tvec> axyz_ext_field_refs = {};
    part_counts->indexes                          = {};

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
        part_counts->indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
        xyz_field_refs.add_obj(p.id_patch, std::ref(pdat.get_field<Tvec>(ixyz)));
        axyz_ext_field_refs.add_obj(p.id_patch, std::ref(pdat.get_field<Tvec>(iaxyz_ext)));
    });

    xyz_refs->set_refs(xyz_field_refs);
    axyz_ext_refs->set_refs(axyz_ext_field_refs);

    auto constant_G  = IDataEdge<Tscal>::make_shared("constant_G", "G");
    auto gpart_mass  = IDataEdge<Tscal>::make_shared("gpart_mass", "m_{\\rm part}");
    constant_G->data = solver_config.get_constant_G();
    gpart_mass->data = solver_config.gpart_mass;

    // sink edges
    auto sink_pos
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>("sink_pos");
    auto sink_mass
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tscal>>>("sink_mass");
    auto sink_accr_radii = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tscal>>>(
        "sink_accretion_radius");
    auto sink_acc_sph
        = sync.template get_edge_ptr<IDataEdgeSerializable<std::vector<Tvec>>>("sink_acc_sph");

    SinkParticlesAddSPHForces<Tvec> add_sph_forces{};
    add_sph_forces.set_edges(
        constant_G,
        gpart_mass,
        part_counts,
        xyz_refs,
        sink_pos,
        sink_mass,
        sink_accr_radii,
        axyz_ext_refs,
        sink_acc_sph);
    add_sph_forces.evaluate();
}

using namespace shammath;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M6>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, M8>;

template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C2>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C4>;
template class shammodels::sph::modules::SinkParticlesUpdate<f64_3, C6>;
