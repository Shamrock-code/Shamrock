// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DragIntegrator.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammodels/ramses/modules/ApplyDragExpo.hpp"
#include "shammodels/ramses/modules/ApplyDragIRK1.hpp"
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shammodels/ramses/modules/ForwardEulerDust.hpp"
#include "shammodels/ramses/modules/ForwardEulerGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::involve_with_no_src(
    Tscal dt) {

    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    using namespace shamrock::solvergraph;

    const u32 ndust = solver_config.dust_config.ndust;

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    storage.rho_next_no_drag = std::make_shared<Field<Tscal>>(
        AMRBlock::block_size, "rho_next_bf_drag", "\\rho^{n+1}_{\\rm no\\,drag}");
    storage.rhov_next_no_drag = std::make_shared<Field<Tvec>>(
        AMRBlock::block_size, "rhov_next_bf_drag", "(\\rho \\mathbf{v})^{n+1}_{\\rm no\\,drag}");
    storage.rhoe_next_no_drag = std::make_shared<Field<Tscal>>(
        AMRBlock::block_size, "rhoe_next_bf_drag", "(\\rho E)^{n+1}_{\\rm no\\,drag}");
    storage.rho_d_next_no_drag = std::make_shared<Field<Tscal>>(
        ndust * AMRBlock::block_size, "rho_d_next_bf_drag", "\\rho^{n+1}_{\\rm dust,\\,no\\,drag}");
    storage.rhov_d_next_no_drag = std::make_shared<Field<Tvec>>(
        ndust * AMRBlock::block_size,
        "rhov_d_next_bf_drag",
        "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})^{n+1}_{\\rm no\\,drag}");

    auto dt_edge   = std::make_shared<ScalarEdge<Tscal>>("dt", "\\Delta t");
    dt_edge->value = dt;

    std::vector<std::shared_ptr<INode>> sequence;

    // U^{n+1} starts as a copy of the state at t^n (patch data without ghosts), then gets the
    // forward Euler update in place
    auto copy_source_field = [&]<class T>(
                                 const std::string &field_name,
                                 const std::string &label,
                                 const std::string &tex,
                                 const std::shared_ptr<Field<T>> &target) {
        auto refs = std::make_shared<FieldRefs<T>>(label, tex);

        GetFieldRefFromLayer<T> attach{pdl, field_name};
        attach.set_edges(storage.source_patches, refs);
        sequence.push_back(std::make_shared<decltype(attach)>(std::move(attach)));

        CopyPatchDataField<T> copy{};
        copy.set_edges(refs, target);
        sequence.push_back(std::make_shared<decltype(copy)>(std::move(copy)));
    };

    copy_source_field("rho", "rho_source", "\\rho", storage.rho_next_no_drag);
    copy_source_field("rhovel", "rhovel_source", "(\\rho \\mathbf{v})", storage.rhov_next_no_drag);
    copy_source_field("rhoetot", "rhoetot_source", "(\\rho E)", storage.rhoe_next_no_drag);
    copy_source_field(
        "rho_dust", "rho_dust_source", "\\rho_{\\rm dust}", storage.rho_d_next_no_drag);
    copy_source_field(
        "rhovel_dust",
        "rhovel_dust_source",
        "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})",
        storage.rhov_d_next_no_drag);

    {
        ForwardEulerGas<Tvec> node{AMRBlock::block_size};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho,
            storage.dtrhov,
            storage.dtrhoe,
            storage.rho_next_no_drag,
            storage.rhov_next_no_drag,
            storage.rhoe_next_no_drag);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    {
        ForwardEulerDust<Tvec> node{AMRBlock::block_size, ndust};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho_dust,
            storage.dtrhov_dust,
            storage.rho_d_next_no_drag,
            storage.rhov_d_next_no_drag);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Forward Euler before drag", std::move(sequence));
    seq.evaluate();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_irk1_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::solvergraph;

    const u32 ndust = solver_config.dust_config.ndust;

    // load layout info
    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl_old();

    auto dt_edge   = std::make_shared<ScalarEdge<Tscal>>("dt", "\\Delta t");
    dt_edge->value = dt;

    std::vector<std::shared_ptr<INode>> sequence;

    // refs to the fields of the patch data without ghosts (the ones that are integrated)
    auto attach_source_field
        = [&]<class T>(std::shared_ptr<FieldRefs<T>> &refs, const std::string &field_name) {
              GetFieldRefFromLayer<T> attach{pdl, field_name};
              attach.set_edges(storage.source_patches, refs);
              sequence.push_back(std::make_shared<decltype(attach)>(std::move(attach)));
          };

    auto refs_rho      = std::make_shared<FieldRefs<Tscal>>("rho_source", "\\rho");
    auto refs_rhov     = std::make_shared<FieldRefs<Tvec>>("rhovel_source", "(\\rho \\mathbf{v})");
    auto refs_rhoe     = std::make_shared<FieldRefs<Tscal>>("rhoetot_source", "(\\rho E)");
    auto refs_rho_dust = std::make_shared<FieldRefs<Tscal>>("rho_dust_source", "\\rho_{\\rm dust}");
    auto refs_rhov_dust = std::make_shared<FieldRefs<Tvec>>(
        "rhovel_dust_source", "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})");

    attach_source_field(refs_rho, "rho");
    attach_source_field(refs_rhov, "rhovel");
    attach_source_field(refs_rhoe, "rhoetot");
    attach_source_field(refs_rho_dust, "rho_dust");
    attach_source_field(refs_rhov_dust, "rhovel_dust");

    {
        // alphas are dust collision rates
        const auto &alphas = solver_config.drag_config.alphas;

        ApplyDragIRK1<Tvec> node{
            AMRBlock::block_size,
            ndust,
            std::vector<Tscal>(alphas.begin(), alphas.end()),
            solver_config.drag_config.enable_frictional_heating};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.rho_next_no_drag,
            storage.rhov_next_no_drag,
            storage.rhoe_next_no_drag,
            storage.rho_d_next_no_drag,
            storage.rhov_d_next_no_drag,
            refs_rho,
            refs_rhov,
            refs_rhoe,
            refs_rho_dust,
            refs_rhov_dust);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Drag IRK1", std::move(sequence));
    seq.evaluate();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_expo_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::solvergraph;

    const u32 ndust = solver_config.dust_config.ndust;

    // load layout info
    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl_old();

    auto dt_edge   = std::make_shared<ScalarEdge<Tscal>>("dt", "\\Delta t");
    dt_edge->value = dt;

    std::vector<std::shared_ptr<INode>> sequence;

    // refs to the fields of the patch data without ghosts (the ones that are integrated)
    auto attach_source_field
        = [&]<class T>(std::shared_ptr<FieldRefs<T>> &refs, const std::string &field_name) {
              GetFieldRefFromLayer<T> attach{pdl, field_name};
              attach.set_edges(storage.source_patches, refs);
              sequence.push_back(std::make_shared<decltype(attach)>(std::move(attach)));
          };

    auto refs_rho      = std::make_shared<FieldRefs<Tscal>>("rho_source", "\\rho");
    auto refs_rhov     = std::make_shared<FieldRefs<Tvec>>("rhovel_source", "(\\rho \\mathbf{v})");
    auto refs_rhoe     = std::make_shared<FieldRefs<Tscal>>("rhoetot_source", "(\\rho E)");
    auto refs_rho_dust = std::make_shared<FieldRefs<Tscal>>("rho_dust_source", "\\rho_{\\rm dust}");
    auto refs_rhov_dust = std::make_shared<FieldRefs<Tvec>>(
        "rhovel_dust_source", "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})");

    attach_source_field(refs_rho, "rho");
    attach_source_field(refs_rhov, "rhovel");
    attach_source_field(refs_rhoe, "rhoetot");
    attach_source_field(refs_rho_dust, "rho_dust");
    attach_source_field(refs_rhov_dust, "rhovel_dust");

    {
        // alphas are dust collision rates
        const auto &alphas = solver_config.drag_config.alphas;

        ApplyDragExpo<Tvec> node{
            AMRBlock::block_size,
            ndust,
            std::vector<Tscal>(alphas.begin(), alphas.end()),
            solver_config.drag_config.enable_frictional_heating};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.rho_next_no_drag,
            storage.rhov_next_no_drag,
            storage.rhoe_next_no_drag,
            storage.rho_d_next_no_drag,
            storage.rhov_d_next_no_drag,
            refs_rho,
            refs_rhov,
            refs_rhoe,
            refs_rho_dust,
            refs_rhov_dust);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Drag expo", std::move(sequence));
    seq.evaluate();
}

template class shammodels::basegodunov::modules::DragIntegrator<f64_3, i64_3>;
