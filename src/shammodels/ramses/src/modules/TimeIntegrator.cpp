// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TimeIntegrator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shammodels/ramses/modules/TimeIntegrator.hpp"
#include "shammodels/ramses/modules/ForwardEulerDust.hpp"
#include "shammodels/ramses/modules/ForwardEulerGas.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include <memory>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegrator<Tvec, TgridVec>::forward_euler(Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::solvergraph;

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

    {
        auto refs_rho  = std::make_shared<FieldRefs<Tscal>>("rho_source", "\\rho");
        auto refs_rhov = std::make_shared<FieldRefs<Tvec>>("rhovel_source", "(\\rho \\mathbf{v})");
        auto refs_rhoe = std::make_shared<FieldRefs<Tscal>>("rhoetot_source", "(\\rho E)");

        attach_source_field(refs_rho, "rho");
        attach_source_field(refs_rhov, "rhovel");
        attach_source_field(refs_rhoe, "rhoetot");

        ForwardEulerGas<Tvec> node{AMRBlock::block_size};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho,
            storage.dtrhov,
            storage.dtrhoe,
            refs_rho,
            refs_rhov,
            refs_rhoe);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    if (solver_config.is_dust_on()) {
        auto refs_rho_dust
            = std::make_shared<FieldRefs<Tscal>>("rho_dust_source", "\\rho_{\\rm dust}");
        auto refs_rhov_dust = std::make_shared<FieldRefs<Tvec>>(
            "rhovel_dust_source", "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})");

        attach_source_field(refs_rho_dust, "rho_dust");
        attach_source_field(refs_rhov_dust, "rhovel_dust");

        ForwardEulerDust<Tvec> node{AMRBlock::block_size, solver_config.dust_config.ndust};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho_dust,
            storage.dtrhov_dust,
            refs_rho_dust,
            refs_rhov_dust);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Forward Euler", std::move(sequence));
    seq.evaluate();
}

template class shammodels::basegodunov::modules::TimeIntegrator<f64_3, i64_3>;
