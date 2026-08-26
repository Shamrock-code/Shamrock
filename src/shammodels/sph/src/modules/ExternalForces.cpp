// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExternalForces.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/common/modules/AddForceCentralGravPotential.hpp"
#include "shammodels/common/modules/AddForceLenseThirring.hpp"
#include "shammodels/common/modules/AddForcePaczynskiWiita.hpp"
#include "shammodels/common/modules/AddForceShearingBoxInertialPart.hpp"
#include "shammodels/common/modules/AddForceShearingBoxNonInertial.hpp"
#include "shammodels/common/modules/AddForceVelocityDissipation.hpp"
#include "shammodels/common/modules/AddForceVerticalDiscPotential.hpp"
#include "shammodels/sph/modules/ExternalForces.hpp"
#include "shammodels/sph/modules/SinkParticlesEvictAccretedParticles.hpp"
#include "shammodels/sph/modules/SinkParticlesFlagAccreteHard.hpp"
#include "shammodels/sph/modules/SinkParticlesUpdate.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"
#include "shamsolvergraph/SolverGraph.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamsolvergraph/node/NodeFreeAlloc.hpp"
#include "shamsolvergraph/node/NodeSetEdge.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamunits/Constants.hpp"

namespace shambase {

    template<class T>
    std::shared_ptr<T> to_shared(T &&t) {
        return std::make_shared<T>(std::forward<T>(t));
    }
} // namespace shambase

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::compute_ext_forces_indep_v() {

    StackEntry stack_loc{};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");
    modules::SinkParticlesUpdate<Tvec, SPHKernel> sink_update(context, solver_config, storage);

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        PatchDataField<Tvec> &field = pdat.get_field<Tvec>(iaxyz_ext);
        field.field_raz();
    });

    sink_update.compute_sph_forces();

    if (solver_config.ext_force_config.ext_forces.empty()) {
        return;
    }

    auto field_xyz = shamrock::solvergraph::FieldRefs<Tvec>::make_shared("", "");

    shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::FieldRefs<Tvec>> set_field_xyz(
        [&](shamrock::solvergraph::FieldRefs<Tvec> &field_xyz_edge) {
            shamrock::solvergraph::DDPatchDataFieldRef<Tvec> field_xyz_refs = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto &field = pdat.get_field<Tvec>(0);
                field_xyz_refs.add_obj(p.id_patch, std::ref(field));
            });
            field_xyz_edge.set_refs(field_xyz_refs);
        });
    set_field_xyz.set_edges(field_xyz);
    set_field_xyz.evaluate();

    auto field_axyz_ext = shamrock::solvergraph::FieldRefs<Tvec>::make_shared("", "");

    shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::FieldRefs<Tvec>> set_field_axyz_ext(
        [&](shamrock::solvergraph::FieldRefs<Tvec> &field_axyz_ext_edge) {
            shamrock::solvergraph::DDPatchDataFieldRef<Tvec> field_axyz_ext_refs = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto &field = pdat.get_field<Tvec>(iaxyz_ext);
                field_axyz_ext_refs.add_obj(p.id_patch, std::ref(field));
            });
            field_axyz_ext_edge.set_refs(field_axyz_ext_refs);
        });
    set_field_axyz_ext.set_edges(field_axyz_ext);
    set_field_axyz_ext.evaluate();

    auto sizes = shamrock::solvergraph::Indexes<u32>::make_shared("", "");

    shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::Indexes<u32>> set_sizes(
        [&](shamrock::solvergraph::Indexes<u32> &sizes) {
            sizes.indexes = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                sizes.indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
            });
        });
    set_sizes.set_edges(sizes);
    set_sizes.evaluate();

    auto constant_G = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
    auto constant_c = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

    shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_constant_G(
        [&](shamrock::solvergraph::IDataEdge<Tscal> &constant_G) {
            constant_G.data = solver_config.get_constant_G();
        });

    shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_constant_c(
        [&](shamrock::solvergraph::IDataEdge<Tscal> &constant_c) {
            constant_c.data = solver_config.get_constant_c();
        });

    set_constant_G.set_edges(constant_G);
    set_constant_c.set_edges(constant_c);

    std::vector<std::shared_ptr<shamrock::solvergraph::INode>> add_ext_forces_seq{};

    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {

            auto central_mass = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
            auto central_pos  = shamrock::solvergraph::IDataEdge<Tvec>::make_shared("", "");

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>>
                set_central_mass([cmass = ext_force->central_mass](
                                     shamrock::solvergraph::IDataEdge<Tscal> &central_mass) {
                    central_mass.data = cmass;
                });
            set_central_mass.set_edges(central_mass);

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tvec>>
                set_central_pos([&](shamrock::solvergraph::IDataEdge<Tvec> &central_pos) {
                    central_pos.data = {}; // no support for offset yet
                });
            set_central_pos.set_edges(central_pos);

            common::modules::AddForceCentralGravPotential<Tvec> add_force_central_grav_potential;
            add_force_central_grav_potential.set_edges(
                constant_G, central_mass, central_pos, field_xyz, sizes, field_axyz_ext);

            add_ext_forces_seq.push_back(
                std::make_shared<shamrock::solvergraph::OperationSequence>(
                    "Point mass",
                    std::vector<std::shared_ptr<shamrock::solvergraph::INode>>{
                        shambase::to_shared(std::move(set_central_pos)),
                        shambase::to_shared(std::move(set_central_mass)),
                        shambase::to_shared(std::move(add_force_central_grav_potential))}));

        } else if (EF_PN_PW *ext_force = std::get_if<EF_PN_PW>(&var_force.val)) {

            auto central_mass = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
            auto central_pos  = shamrock::solvergraph::IDataEdge<Tvec>::make_shared("", "");

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>>
                set_central_mass([cmass = ext_force->central_mass](
                                     shamrock::solvergraph::IDataEdge<Tscal> &central_mass) {
                    central_mass.data = cmass;
                });
            set_central_mass.set_edges(central_mass);

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tvec>>
                set_central_pos([cpos = ext_force->central_pos](
                                    shamrock::solvergraph::IDataEdge<Tvec> &central_pos) {
                    central_pos.data = cpos;
                });
            set_central_pos.set_edges(central_pos);

            common::modules::AddForcePaczynskiWiita<Tvec> add_force_paczynski_wiita;
            add_force_paczynski_wiita.set_edges(
                constant_G,
                constant_c,
                central_mass,
                central_pos,
                field_xyz,
                sizes,
                field_axyz_ext);

            add_ext_forces_seq.push_back(
                std::make_shared<shamrock::solvergraph::OperationSequence>(
                    "Pseudo-Newtonian PW",
                    std::vector<std::shared_ptr<shamrock::solvergraph::INode>>{
                        shambase::to_shared(std::move(set_central_pos)),
                        shambase::to_shared(std::move(set_central_mass)),
                        shambase::to_shared(std::move(add_force_paczynski_wiita))}));

        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {

            auto central_mass = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
            auto central_pos  = shamrock::solvergraph::IDataEdge<Tvec>::make_shared("", "");

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>>
                set_central_mass([cmass = ext_force->central_mass](
                                     shamrock::solvergraph::IDataEdge<Tscal> &central_mass) {
                    central_mass.data = cmass;
                });
            set_central_mass.set_edges(central_mass);

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tvec>>
                set_central_pos([&](shamrock::solvergraph::IDataEdge<Tvec> &central_pos) {
                    central_pos.data = {}; // no support for offset yet
                });
            set_central_pos.set_edges(central_pos);

            common::modules::AddForceCentralGravPotential<Tvec> add_force_central_grav_potential;
            add_force_central_grav_potential.set_edges(
                constant_G, central_mass, central_pos, field_xyz, sizes, field_axyz_ext);

            add_ext_forces_seq.push_back(
                std::make_shared<shamrock::solvergraph::OperationSequence>(
                    "Point mass",
                    std::vector<std::shared_ptr<shamrock::solvergraph::INode>>{
                        shambase::to_shared(std::move(set_central_pos)),
                        shambase::to_shared(std::move(set_central_mass)),
                        shambase::to_shared(std::move(add_force_central_grav_potential))}));

        } else if (
            EF_ShearingBoxForce *ext_force = std::get_if<EF_ShearingBoxForce>(&var_force.val)) {

            auto eta = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_eta(
                [&](shamrock::solvergraph::IDataEdge<Tscal> &eta) {
                    eta.data = ext_force->eta;
                });
            set_eta.set_edges(eta);

            common::modules::AddForceShearingBoxInertialPart<Tvec>
                add_force_shearing_box_inertial_part{};
            add_force_shearing_box_inertial_part.set_edges(eta, field_xyz, sizes, field_axyz_ext);

            add_ext_forces_seq.push_back(
                std::make_shared<shamrock::solvergraph::OperationSequence>(
                    "Shearing box force",
                    std::vector<std::shared_ptr<shamrock::solvergraph::INode>>{
                        shambase::to_shared(std::move(set_eta)),
                        shambase::to_shared(std::move(add_force_shearing_box_inertial_part))}));

        } else if (
            EF_VerticalDiscPotential *ext_force
            = std::get_if<EF_VerticalDiscPotential>(&var_force.val)) {

            auto central_mass = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
            auto R0           = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>>
                set_central_mass([cmass = ext_force->central_mass](
                                     shamrock::solvergraph::IDataEdge<Tscal> &central_mass) {
                    central_mass.data = cmass;
                });
            set_central_mass.set_edges(central_mass);

            shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<Tscal>> set_R0(
                [r = ext_force->R0](shamrock::solvergraph::IDataEdge<Tscal> &R0) {
                    R0.data = r; // no support for offset yet
                });
            set_R0.set_edges(R0);

            common::modules::AddForceVerticalDiscPotential<Tvec> add_force_vertical_disc_potential;
            add_force_vertical_disc_potential.set_edges(
                constant_G, central_mass, R0, field_xyz, sizes, field_axyz_ext);

            add_ext_forces_seq.push_back(
                std::make_shared<shamrock::solvergraph::OperationSequence>(
                    "Vertical disc potential",
                    std::vector<std::shared_ptr<shamrock::solvergraph::INode>>{
                        shambase::to_shared(std::move(set_R0)),
                        shambase::to_shared(std::move(set_central_mass)),
                        shambase::to_shared(std::move(add_force_vertical_disc_potential))}));

        } else if (
            EF_VelocityDissipation *ext_force
            = std::get_if<EF_VelocityDissipation>(&var_force.val)) {

        } else {
            shambase::throw_unimplemented("this force is not handled, yet ...");
        }
    }

    set_constant_G.evaluate();
    set_constant_c.evaluate();

    if (add_ext_forces_seq.size() > 0) {
        shamrock::solvergraph::OperationSequence seq(
            "Add external forces", std::move(add_ext_forces_seq));
        seq.evaluate();
    }
}

template<class T>
std::shared_ptr<shamrock::solvergraph::INode> register_constant_set(
    shamrock::solvergraph::SolverGraph &solver_graph, std::string name, std::function<T()> getter) {
    solver_graph.register_edge(name, shamrock::solvergraph::IDataEdge<T>("", ""));

    solver_graph.register_node(
        "set_" + name,
        shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<T>>(
            [getter](shamrock::solvergraph::IDataEdge<T> &edge) {
                edge.data = getter();
            }));

    solver_graph
        .get_node_ref<shamrock::solvergraph::NodeSetEdge<shamrock::solvergraph::IDataEdge<T>>>(
            "set_" + name)
        .set_edges(solver_graph.get_edge_ptr_base(name));

    return solver_graph.get_node_ptr_base("set_" + name);
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::add_ext_forces() {

    StackEntry stack_loc{};

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tscal gpart_mass = solver_config.gpart_mass;

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 iaxyz     = pdl.get_field_idx<Tvec>("axyz");
    const u32 ivxyz     = pdl.get_field_idx<Tvec>("vxyz");
    const u32 iaxyz_ext = pdl.get_field_idx<Tvec>("axyz_ext");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceBuffer<Tvec> &buf_axyz     = pdat.get_field_buf_ref<Tvec>(iaxyz);
        sham::DeviceBuffer<Tvec> &buf_axyz_ext = pdat.get_field_buf_ref<Tvec>(iaxyz_ext);

        sham::EventList depends_list;
        auto axyz     = buf_axyz.get_write_access(depends_list);
        auto axyz_ext = buf_axyz_ext.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shambase::parallel_for(
                cgh, pdat.get_obj_cnt(), "add ext force acc to acc", [=](u64 gid) {
                    axyz[gid] += axyz_ext[gid];
                });
        });

        buf_axyz.complete_event_state(e);
        buf_axyz_ext.complete_event_state(e);
    });

    if (solver_config.ext_force_config.ext_forces.empty()) {
        return; // skip if no external forces
    }

    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_PN_PW             = typename SolverConfigExtForce::PN_PW;
    using EF_LenseThirring     = typename SolverConfigExtForce::LenseThirring;

    using namespace shamrock::solvergraph;
    SolverGraph solver_graph{};

    auto set_constant_G = register_constant_set<Tscal>(solver_graph, "constant_G", [&]() {
        return solver_config.get_constant_G();
    });
    auto set_constant_c = register_constant_set<Tscal>(solver_graph, "constant_c", [&]() {
        return solver_config.get_constant_c();
    });

    bool is_G_needed = false;
    bool is_c_needed = false;

    for (auto var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {

        } else if (EF_PN_PW *ext_force = std::get_if<EF_PN_PW>(&var_force.val)) {
            is_G_needed = true;
            is_c_needed = true;
        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {
            is_G_needed = true;
            is_c_needed = true;
        } else if (
            EF_ShearingBoxForce *ext_force = std::get_if<EF_ShearingBoxForce>(&var_force.val)) {
        } else if (
            EF_VerticalDiscPotential *ext_force
            = std::get_if<EF_VerticalDiscPotential>(&var_force.val)) {
        } else if (
            EF_VelocityDissipation *ext_force
            = std::get_if<EF_VelocityDissipation>(&var_force.val)) {
        } else {
            shambase::throw_unimplemented("this force is not handled, yet ...");
        }
    }

    std::vector<std::shared_ptr<shamrock::solvergraph::INode>> add_ext_forces_seq{};

    if (is_G_needed) {
        add_ext_forces_seq.push_back(set_constant_G);
    }
    if (is_c_needed) {
        add_ext_forces_seq.push_back(set_constant_c);
    }

    auto field_xyz   = solver_graph.register_edge("field_xyz", FieldRefs<Tvec>("", ""));
    auto field_vxyz  = solver_graph.register_edge("field_vxyz", FieldRefs<Tvec>("", ""));
    auto field_axyz  = solver_graph.register_edge("field_axyz", FieldRefs<Tvec>("", ""));
    auto field_sizes = solver_graph.register_edge("field_sizes", Indexes<u32>("", ""));

    auto set_field_xyz = solver_graph.register_node(
        "set_field_xyz", NodeSetEdge<FieldRefs<Tvec>>([&](FieldRefs<Tvec> &field_xyz_edge) {
            DDPatchDataFieldRef<Tvec> field_xyz_refs = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto &field = pdat.get_field<Tvec>(0);
                field_xyz_refs.add_obj(p.id_patch, std::ref(field));
            });
            field_xyz_edge.set_refs(field_xyz_refs);
        }));
    shambase::get_check_ref(set_field_xyz).set_edges(field_xyz);

    auto set_field_vxyz = solver_graph.register_node(
        "set_field_vxyz", NodeSetEdge<FieldRefs<Tvec>>([&](FieldRefs<Tvec> &field_vxyz_edge) {
            DDPatchDataFieldRef<Tvec> field_vxyz_refs = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto &field = pdat.get_field<Tvec>(ivxyz);
                field_vxyz_refs.add_obj(p.id_patch, std::ref(field));
            });
            field_vxyz_edge.set_refs(field_vxyz_refs);
        }));
    shambase::get_check_ref(set_field_vxyz).set_edges(field_vxyz);

    auto set_field_axyz = solver_graph.register_node(
        "set_field_axyz", NodeSetEdge<FieldRefs<Tvec>>([&](FieldRefs<Tvec> &field_axyz_edge) {
            DDPatchDataFieldRef<Tvec> field_axyz_refs = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                auto &field = pdat.get_field<Tvec>(iaxyz);
                field_axyz_refs.add_obj(p.id_patch, std::ref(field));
            });
            field_axyz_edge.set_refs(field_axyz_refs);
        }));
    shambase::get_check_ref(set_field_axyz).set_edges(field_axyz);

    auto set_field_sizes = solver_graph.register_node(
        "set_field_sizes", NodeSetEdge<Indexes<u32>>([&](Indexes<u32> &sizes) {
            sizes.indexes = {};
            scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                sizes.indexes.add_obj(p.id_patch, pdat.get_obj_cnt());
            });
        }));
    shambase::get_check_ref(set_field_sizes).set_edges(field_sizes);

    add_ext_forces_seq.push_back(set_field_xyz);
    add_ext_forces_seq.push_back(set_field_vxyz);
    add_ext_forces_seq.push_back(set_field_axyz);
    add_ext_forces_seq.push_back(set_field_sizes);

    for (u32 i = 0; i < solver_config.ext_force_config.ext_forces.size(); i++) {

        auto &var_force = solver_config.ext_force_config.ext_forces[i];

        std::string prefix = sham::format("ext_force_{}_", i);

        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {

        } else if (EF_PN_PW *ext_force = std::get_if<EF_PN_PW>(&var_force.val)) {

        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {

            std::string prefix_cmass       = prefix + "cmass_";
            std::string prefix_central_pos = prefix + "central_pos_";
            std::string prefix_a_spin      = prefix + "a_spin_";
            std::string prefix_dir_spin    = prefix + "dir_spin_";
            std::string prefix_lt          = prefix + "lt_";

            auto set_cmass = register_constant_set<Tscal>(solver_graph, prefix_cmass, [&]() {
                return ext_force->central_mass;
            });

            auto set_central_pos
                = register_constant_set<Tvec>(solver_graph, prefix_central_pos, [&]() {
                      return Tvec{0, 0, 0}; // no support for offset yet
                  });

            auto set_a_spin = register_constant_set<Tscal>(solver_graph, prefix_a_spin, [&]() {
                return ext_force->a_spin;
            });

            auto set_dir_spin = register_constant_set<Tvec>(solver_graph, prefix_dir_spin, [&]() {
                return ext_force->dir_spin;
            });

            auto add_force_lense_thirring = solver_graph.register_node(
                prefix_lt, shammodels::common::modules::AddForceLenseThirring<Tvec>());
            shambase::get_check_ref(add_force_lense_thirring)
                .set_edges(
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>("constant_G"),
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>("constant_c"),
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>(prefix_cmass),
                    solver_graph.get_edge_ptr<IDataEdge<Tvec>>(prefix_central_pos),
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>(prefix_a_spin),
                    solver_graph.get_edge_ptr<IDataEdge<Tvec>>(prefix_dir_spin),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_xyz"),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_vxyz"),
                    solver_graph.get_edge_ptr<Indexes<u32>>("field_sizes"),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_axyz"));

            add_ext_forces_seq.push_back(set_cmass);
            add_ext_forces_seq.push_back(set_central_pos);
            add_ext_forces_seq.push_back(set_a_spin);
            add_ext_forces_seq.push_back(set_dir_spin);
            add_ext_forces_seq.push_back(solver_graph.get_node_ptr_base(prefix_lt));

        } else if (
            EF_ShearingBoxForce *ext_force = std::get_if<EF_ShearingBoxForce>(&var_force.val)) {

            std::string prefix_Omega_0      = prefix + "Omega_0_";
            std::string prefix_q            = prefix + "q_";
            std::string prefix_shearing_box = prefix + "shearing_box_";

            auto set_Omega_0 = register_constant_set<Tscal>(solver_graph, prefix_Omega_0, [&]() {
                return ext_force->Omega_0;
            });

            auto set_q = register_constant_set<Tscal>(solver_graph, prefix_q, [&]() {
                return ext_force->q;
            });

            auto add_force_shearing_box_non_inertial = solver_graph.register_node(
                prefix_shearing_box,
                shammodels::common::modules::AddForceShearingBoxNonInertial<Tvec>());
            shambase::get_check_ref(add_force_shearing_box_non_inertial)
                .set_edges(
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>(prefix_Omega_0),
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>(prefix_q),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_xyz"),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_vxyz"),
                    solver_graph.get_edge_ptr<Indexes<u32>>("field_sizes"),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_axyz"));

            add_ext_forces_seq.push_back(set_Omega_0);
            add_ext_forces_seq.push_back(set_q);
            add_ext_forces_seq.push_back(solver_graph.get_node_ptr_base(prefix_shearing_box));

        } else if (
            EF_VerticalDiscPotential *ext_force
            = std::get_if<EF_VerticalDiscPotential>(&var_force.val)) {
        } else if (
            EF_VelocityDissipation *ext_force
            = std::get_if<EF_VelocityDissipation>(&var_force.val)) {
            std::string prefix_eta                  = prefix + "eta_";
            std::string prefix_velocity_dissipation = prefix + "velocity_dissipation_";

            auto set_eta
                = register_constant_set<Tscal>(solver_graph, prefix_eta, [eta = ext_force->eta]() {
                      return eta;
                  });

            auto add_force_velocity_dissipation = solver_graph.register_node(
                prefix_velocity_dissipation,
                shammodels::common::modules::AddForceVelocityDissipation<Tvec>());
            shambase::get_check_ref(add_force_velocity_dissipation)
                .set_edges(
                    solver_graph.get_edge_ptr<IDataEdge<Tscal>>(prefix_eta),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_vxyz"),
                    solver_graph.get_edge_ptr<Indexes<u32>>("field_sizes"),
                    solver_graph.get_edge_ptr<IFieldSpan<Tvec>>("field_axyz"));

            add_ext_forces_seq.push_back(set_eta);
            add_ext_forces_seq.push_back(
                solver_graph.get_node_ptr_base(prefix_velocity_dissipation));

        } else {
            shambase::throw_unimplemented("this force is not handled, yet ...");
        }
    }

    if (add_ext_forces_seq.size() > 0) {
        OperationSequence seq("Add external forces", std::move(add_ext_forces_seq));
        seq.evaluate();
    }
}

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ExternalForces<Tvec, SPHKernel>::point_mass_accrete_particles() {

    StackEntry stack_loc{};

    using namespace shamrock;
    using namespace shamrock::patch;
    using namespace shamrock::solvergraph;

    using SolverConfigExtForce = typename Config::ExtForceConfig;
    using EF_PointMass         = typename SolverConfigExtForce::PointMass;
    using EF_LenseThirring     = typename SolverConfigExtForce::LenseThirring;

    SolverGraph &solver_graph = storage.solver_graph;

    if (!solver_graph.has_edge("ext_force_accretion_pos")) {
        solver_graph.register_edge("ext_force_accretion_pos", IDataEdge<std::vector<Tvec>>("", ""));
    }
    if (!solver_graph.has_edge("ext_force_accretion_racc")) {
        solver_graph.register_edge(
            "ext_force_accretion_racc", IDataEdge<std::vector<Tscal>>("", ""));
    }

    auto accretion_positions
        = solver_graph.get_edge_ptr<IDataEdge<std::vector<Tvec>>>("ext_force_accretion_pos");
    auto accretion_radii
        = solver_graph.get_edge_ptr<IDataEdge<std::vector<Tscal>>>("ext_force_accretion_racc");

    accretion_positions->data.clear();
    accretion_radii->data.clear();
    for (auto &var_force : solver_config.ext_force_config.ext_forces) {
        if (EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {
            accretion_positions->data.push_back({0, 0, 0}); // no support for offset yet
            accretion_radii->data.push_back(ext_force->Racc);
        } else if (EF_PN_PW *ext_force = std::get_if<EF_PN_PW>(&var_force.val)) {
            accretion_positions->data.push_back({0, 0, 0}); // no support for offset yet
            accretion_radii->data.push_back(ext_force->Racc);
        } else if (EF_LenseThirring *ext_force = std::get_if<EF_LenseThirring>(&var_force.val)) {
            accretion_positions->data.push_back({0, 0, 0}); // no support for offset yet
            accretion_radii->data.push_back(ext_force->Racc);
        }
    }

    if (accretion_positions->data.empty()) {
        return;
    }

    if (!solver_graph.has_edge("ext_force_accretion_table")) {
        solver_graph.register_edge(
            "ext_force_accretion_table",
            Field<u32>(1, "ext_force_accretion_table", "\\mathrm{acc}"));
    }

    if (!solver_graph.has_node("ext_force_flag_accrete_hard")) {
        auto flag_node = solver_graph.register_node(
            "ext_force_flag_accrete_hard", modules::SinkParticlesFlagAccreteHard<Tvec>{});
        shambase::get_check_ref(flag_node).set_edges(
            solver_graph.get_edge_ptr<Indexes<u32>>("part_counts"),
            solver_graph.get_edge_ptr<FieldRefs<Tvec>>("xyz"),
            accretion_positions,
            accretion_radii,
            solver_graph.get_edge_ptr<Field<u32>>("ext_force_accretion_table"));
    }

    if (!solver_graph.has_node("ext_force_evict_accreted")) {
        auto evict_node = solver_graph.register_node(
            "ext_force_evict_accreted", modules::SinkParticlesEvictAccretedParticles<Tvec>{});
        shambase::get_check_ref(evict_node)
            .set_edges(
                solver_graph.get_edge_ptr<Indexes<u32>>("part_counts"),
                solver_graph.get_edge_ptr<Field<u32>>("ext_force_accretion_table"),
                solver_graph.get_edge_ptr<PatchDataLayerRefs>("scheduler_patchdata"));
    }

    if (!solver_graph.has_node("ext_force_free_xyz_refs")) {
        auto free_xyz = solver_graph.register_node("ext_force_free_xyz_refs", NodeFreeAlloc{});
        shambase::get_check_ref(free_xyz).set_edges(
            solver_graph.get_edge_ptr<FieldRefs<Tvec>>("xyz"));
    }

    OperationSequence(
        "Point mass accretion",
        {
            solver_graph.get_node_ptr_base("set_gpart_mass"),
            solver_graph.get_node_ptr_base("set_scheduler_patchdata"),
            solver_graph.get_node_ptr_base("attach_part_counts"),
            solver_graph.get_node_ptr_base("attach_xyz"),
            solver_graph.get_node_ptr_base("ext_force_flag_accrete_hard"),
            solver_graph.get_node_ptr_base("ext_force_evict_accreted"),
            solver_graph.get_node_ptr_base("ext_force_free_xyz_refs"),
        })
        .evaluate();
}

using namespace shammath;
template class shammodels::sph::modules::ExternalForces<f64_3, M4>;
template class shammodels::sph::modules::ExternalForces<f64_3, M6>;
template class shammodels::sph::modules::ExternalForces<f64_3, M8>;

template class shammodels::sph::modules::ExternalForces<f64_3, C2>;
template class shammodels::sph::modules::ExternalForces<f64_3, C4>;
template class shammodels::sph::modules::ExternalForces<f64_3, C6>;
