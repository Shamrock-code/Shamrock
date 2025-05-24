// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCellInfos.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeCellInfos.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/AMRCellInfos.hpp"
#include "shammodels/ramses/modules/ComputeCellAABB.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ComputeCellInfos<Tvec, TgridVec>::compute_aabb() {

    StackEntry stack_loc{};

    // will be filled by NodeComputeCellAABB
    auto spans_block_cell_sizes = std::make_shared<shamrock::solvergraph::Field<Tscal>>(
        1, "block_cell_sizes", "s_{\\rm cell}");
    auto spans_cell0block_aabb_lower = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        1, "cell0block_aabb_lower", "\\mathbf{s}_{\\rm inf,block}");

    NodeComputeCellAABB<Tvec, TgridVec> node{AMRBlock::Nside, solver_config.grid_coord_to_pos_fact};

    node.set_edges(
        storage.block_counts_with_ghost,
        storage.refs_block_min,
        storage.refs_block_max,
        spans_block_cell_sizes,
        spans_cell0block_aabb_lower);
    node.evaluate();

    // logger::raw_ln(" --- dot:\n" + node.get_dot_graph());
    // logger::raw_ln(" --- tex:\n" + node.get_tex());

    shamrock::ComputeField<Tscal> block_cell_sizes     = spans_block_cell_sizes->extract();
    shamrock::ComputeField<Tvec> cell0block_aabb_lower = spans_cell0block_aabb_lower->extract();

    // logger::raw_ln("block_cell_sizes", ":");
    // block_cell_sizes.field_data.for_each([&](u64 id, PatchDataField<Tscal> &pdf) {
    //     logger::raw_ln(pdf.get_buf().copy_to_stdvec());
    // });
    //
    // logger::raw_ln("cell0block_aabb_lower", ":");
    // cell0block_aabb_lower.field_data.for_each([&](u64 id, PatchDataField<Tvec> &pdf) {
    //    logger::raw_ln(pdf.get_buf().copy_to_stdvec());
    //});

    storage.cell_infos.set(
        CellInfos<Tvec, TgridVec>{std::move(block_cell_sizes), std::move(cell0block_aabb_lower)});
}

template class shammodels::basegodunov::modules::ComputeCellInfos<f64_3, i64_3>;
