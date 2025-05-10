// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/ConsToPrimDust.hpp"
#include "shammodels/ramses/modules/ConsToPrimGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include <memory>
#include <utility>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_gas() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhov_ghost                                = ghost_layout.get_field_idx<Tvec>("rhovel");
    u32 irhoe_ghost                                = ghost_layout.get_field_idx<Tscal>("rhoetot");

    auto block_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        "block_count_with_ghost", "N_{\\rm block, with ghost}");

    block_counts_with_ghost->indexes
        = storage.merged_patchdata_ghost.get().template map<u32>([&](u64 id, MergedPDat &mpdat) {
              return mpdat.total_elements;
          });

    auto spans_rho   = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>("rho", "\\rho");
    spans_rho->spans = storage.merged_patchdata_ghost.get()
                           .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                               [&](u64 id, MergedPDat &mpdat) {
                                   return mpdat.pdat.get_field_pointer_span<Tscal>(irho_ghost);
                               });

    auto spans_rhov
        = std::make_shared<shamrock::solvergraph::FieldSpan<Tvec>>("rhovel", "(\\rho \\mathbf{v})");
    spans_rhov->spans = storage.merged_patchdata_ghost.get()
                            .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                                [&](u64 id, MergedPDat &mpdat) {
                                    return mpdat.pdat.get_field_pointer_span<Tvec>(irhov_ghost);
                                });

    auto spans_rhoe
        = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>("rhoetot", "(\\rho E)");
    spans_rhoe->spans = storage.merged_patchdata_ghost.get()
                            .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                                [&](u64 id, MergedPDat &mpdat) {
                                    return mpdat.pdat.get_field_pointer_span<Tscal>(irhoe_ghost);
                                });

    // will be filled by NodeConsToPrimGas
    auto spans_vel = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "vel", "\\mathbf{v}");
    auto spans_P
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(AMRBlock::block_size, "P", "P");

    NodeConsToPrimGas<Tvec> node{AMRBlock::block_size, solver_config.eos_gamma};

    node.set_edges(block_counts_with_ghost, spans_rho, spans_rhov, spans_rhoe, spans_vel, spans_P);
    node.evaluate();

    // logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    // logger::raw_ln(" --- tex:\n" + node.get_tex());

    auto print_comp_field_state = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.field_data.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f.get_obj_cnt());
        });
    };

    shamrock::ComputeField<Tvec> v_ghost  = std::move(spans_vel->extract());
    shamrock::ComputeField<Tscal> P_ghost = std::move(spans_P->extract());

    // print_comp_field_state("v_ghost", v_ghost);
    // print_comp_field_state("P_ghost", P_ghost);

    storage.vel.set(std::move(v_ghost));
    storage.press.set(std::move(P_ghost));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim_dust() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    u32 ndust                                      = solver_config.dust_config.ndust;
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

    u32 irho_dust_ghost  = ghost_layout.get_field_idx<Tscal>("rho_dust");
    u32 irhov_dust_ghost = ghost_layout.get_field_idx<Tvec>("rhovel_dust");

    auto block_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        "block_count_with_ghost", "N_{\\rm block, with ghost}");

    block_counts_with_ghost->indexes
        = storage.merged_patchdata_ghost.get().template map<u32>([&](u64 id, MergedPDat &mpdat) {
              return mpdat.total_elements;
          });

    auto spans_rho_dust = std::make_shared<shamrock::solvergraph::FieldSpan<Tscal>>(
        "rho_dust", "\\rho_{\\rm dust}");
    spans_rho_dust->spans
        = storage.merged_patchdata_ghost.get()
              .template map<shamrock::PatchDataFieldSpanPointer<Tscal>>(
                  [&](u64 id, MergedPDat &mpdat) {
                      return mpdat.pdat.get_field_pointer_span<Tscal>(irho_dust_ghost);
                  });

    auto spans_rhov_dust = std::make_shared<shamrock::solvergraph::FieldSpan<Tvec>>(
        "rhovel_dust", "(\\rho_{\\rm dust} \\mathbf{v})_{\\rm dust}");
    spans_rhov_dust->spans
        = storage.merged_patchdata_ghost.get()
              .template map<shamrock::PatchDataFieldSpanPointer<Tvec>>(
                  [&](u64 id, MergedPDat &mpdat) {
                      return mpdat.pdat.get_field_pointer_span<Tvec>(irhov_dust_ghost);
                  });

    // will be filled by NodeConsToPrimDust
    auto spans_vel_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size * ndust, "vel_dust", "\\mathbf{v}_{\\rm dust}");

    auto print_block_sizes = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f);
        });
    };

    // print_block_sizes("block_counts_with_ghost", block_counts_with_ghost->indexes);

    NodeConsToPrimDust<Tvec> node{AMRBlock::block_size, ndust};

    node.set_edges(block_counts_with_ghost, spans_rho_dust, spans_rhov_dust, spans_vel_dust);
    node.evaluate();

    // logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    // logger::raw_ln(" --- tex:\n" + node.get_tex());

    auto print_comp_field_state = [](std::string name, auto &cfield) {
        logger::raw_ln("Comp field state :", name);
        cfield.field_data.for_each([&](u64 id, auto &f) {
            logger::raw_ln(id, f.get_obj_cnt());
        });
    };

    shamrock::ComputeField<Tvec> v_dust_ghost = std::move(spans_vel_dust->extract());

    // print_comp_field_state("v_ghost", v_dust_ghost);

    storage.vel_dust.set(std::move(v_dust_ghost));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_gas() {
    storage.vel.reset();
    storage.press.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::reset_dust() {
    storage.vel_dust.reset();
}

template class shammodels::basegodunov::modules::ConsToPrim<f64_3, i64_3>;
