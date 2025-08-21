// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/timestep_report.hpp"
#include "shammodels/ramses/Solver.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/ramses/modules/BlockNeighToCellNeigh.hpp"
#include "shammodels/ramses/modules/ComputeCFL.hpp"
#include "shammodels/ramses/modules/ComputeCellAABB.hpp"
#include "shammodels/ramses/modules/ComputeMass.hpp"
#include "shammodels/ramses/modules/ComputeSumOverV.hpp"
#include "shammodels/ramses/modules/ComputeTimeDerivative.hpp"
#include "shammodels/ramses/modules/ConsToPrimDust.hpp"
#include "shammodels/ramses/modules/ConsToPrimGas.hpp"
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shammodels/ramses/modules/FindBlockNeigh.hpp"
#include "shammodels/ramses/modules/GhostZones.hpp"
#include "shammodels/ramses/modules/InterpolateToFace.hpp"
#include "shammodels/ramses/modules/NodeComputeFlux.hpp"
#include "shammodels/ramses/modules/SlopeLimitedGradient.hpp"
#include "shammodels/ramses/modules/TimeIntegrator.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/solvergraph/ExtractCounts.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldSpan.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamrock/solvergraph/NodeFreeAlloc.hpp"
#include "shamrock/solvergraph/OperationSequence.hpp"
#include "shamrock/solvergraph/PatchDataLayerEdge.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <memory>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::init_solver_graph() {

    bool enable_mem_free = false;

    auto get_optional_free_mem = [&](auto &bind_to, auto &add_to) {
        if (enable_mem_free) {
            shamrock::solvergraph::NodeFreeAlloc node;
            node.set_edges(bind_to);
            add_to.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }
    };

    {
        storage.ghost_layout = std::make_shared<shamrock::patch::PatchDataLayerLayout>();
        shamrock::patch::PatchDataLayerLayout &ghost_layout
            = shambase::get_check_ref(storage.ghost_layout);

        ghost_layout.add_field<TgridVec>("cell_min", 1);
        ghost_layout.add_field<TgridVec>("cell_max", 1);
        ghost_layout.add_field<Tscal>("rho", AMRBlock::block_size);
        ghost_layout.add_field<Tscal>("rhoetot", AMRBlock::block_size);
        ghost_layout.add_field<Tvec>("rhovel", AMRBlock::block_size);

        if (solver_config.is_dust_on()) {
            auto ndust = solver_config.dust_config.ndust;
            ghost_layout.add_field<Tscal>("rho_dust", ndust * AMRBlock::block_size);
            ghost_layout.add_field<Tvec>("rhovel_dust", ndust * AMRBlock::block_size);
        }

        if (solver_config.is_gravity_on()) {
            ghost_layout.add_field<Tscal>("phi", AMRBlock::block_size);
        }

        if (solver_config.is_gas_passive_scalar_on()) {
            u32 npscal_gas = solver_config.npscal_gas_config.npscal_gas;
            ghost_layout.add_field<Tscal>("rho_gas_pscal", npscal_gas * AMRBlock::block_size);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Edges
    ////////////////////////////////////////////////////////////////////////////////

    storage.patch_rank_owner
        = std::make_shared<shamrock::solvergraph::ScalarsEdge<u32>>("patch_rank_owner", "rank");

    storage.source_patches = std::make_shared<shamrock::solvergraph::PatchDataLayerRefs>(
        "source_patches", "P_{\\rm source}");

    storage.merged_patchdata_ghost = std::make_shared<shamrock::solvergraph::PatchDataLayerEdge>(
        "merged_patchdata_ghost", "patchdata_{\\rm ghost}", storage.ghost_layout);

    storage.block_counts
        = std::make_shared<shamrock::solvergraph::Indexes<u32>>("block_count", "N_{\\rm block}");

    storage.block_counts_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>(
        "block_count_with_ghost", "N_{\\rm block, with ghost}");

    // merged ghost spans
    storage.refs_block_min = std::make_shared<shamrock::solvergraph::FieldRefs<TgridVec>>(
        "block_min", "\\mathbf{r}_{\\rm block, min}");
    storage.refs_block_max = std::make_shared<shamrock::solvergraph::FieldRefs<TgridVec>>(
        "block_max", "\\mathbf{r}_{\\rm block, max}");

    storage.refs_rho = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("rho", "\\rho");
    storage.refs_rhov
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>("rhovel", "(\\rho \\mathbf{v})");
    storage.refs_rhoe
        = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>("rhoetot", "(\\rho E)");

    if (solver_config.is_dust_on()) {
        storage.refs_rho_dust = std::make_shared<shamrock::solvergraph::FieldRefs<Tscal>>(
            "rho_dust", "\\rho_{\\rm dust}");
        storage.refs_rhov_dust = std::make_shared<shamrock::solvergraph::FieldRefs<Tvec>>(
            "rhovel_dust", "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})");
    }

    // will be filled by NodeConsToPrimGas
    storage.vel = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "vel", "\\mathbf{v}");
    storage.press
        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(AMRBlock::block_size, "P", "P");

    if (solver_config.is_dust_on()) {
        u32 ndust = solver_config.dust_config.ndust;

        // will be filled by NodeConsToPrimDust
        storage.vel_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
            AMRBlock::block_size * ndust, "vel_dust", "{\\mathbf{v}_{\\rm dust}}");
    }

    storage.trees
        = std::make_shared<solvergraph::TreeEdge<u_morton, TgridVec>>("trees", "\\text{trees}");

    storage.block_graph_edge = std::make_shared<
        shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(
        "block_graph_edge", "\\text{block graph edge}");

    storage.cell_graph_edge = std::make_shared<
        shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(
        "cell_graph_edge", "\\text{cell graph edge}");

    // will be filled by NodeComputeCellAABB
    storage.block_cell_sizes = std::make_shared<shamrock::solvergraph::Field<Tscal>>(
        1, "block_cell_sizes", "s_{\\rm cell}");
    storage.cell0block_aabb_lower = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        1, "cell0block_aabb_lower", "\\mathbf{s}_{\\rm inf,block}");

    storage.grad_rho = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "grad_rho", "\\nabla \\rho");
    storage.dx_v = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "dx_v", "\\nabla_x \\mathbf{v}");
    storage.dy_v = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "dy_v", "\\nabla_y \\mathbf{v}");
    storage.dz_v = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "dz_v", "\\nabla_z \\mathbf{v}");
    storage.grad_P = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
        AMRBlock::block_size, "grad_P", "\\nabla P");

    if (solver_config.is_dust_on()) {
        u32 ndust             = solver_config.dust_config.ndust;
        storage.grad_rho_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
            AMRBlock::block_size * ndust, "grad_rho_dust", "\\nabla \\rho_{\\rm dust}");
        storage.dx_v_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
            AMRBlock::block_size * ndust, "dx_v_dust", "\\nabla_x \\mathbf{v}_{\\rm dust}");
        storage.dy_v_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
            AMRBlock::block_size * ndust, "dy_v_dust", "\\nabla_y \\mathbf{v}_{\\rm dust}");
        storage.dz_v_dust = std::make_shared<shamrock::solvergraph::Field<Tvec>>(
            AMRBlock::block_size * ndust, "dz_v_dust", "\\nabla_z \\mathbf{v}_{\\rm dust}");
    }

    {

        storage.rho_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_xp", "rho_face_xp", 1);
        storage.rho_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_xm", "rho_face_xm", 1);
        storage.rho_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_yp", "rho_face_yp", 1);
        storage.rho_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_ym", "rho_face_ym", 1);
        storage.rho_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_zp", "rho_face_zp", 1);
        storage.rho_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_face_zm", "rho_face_zm", 1);

        storage.vel_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_xp", "vel_face_xp", 1);
        storage.vel_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_xm", "vel_face_xm", 1);
        storage.vel_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_yp", "vel_face_yp", 1);
        storage.vel_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_ym", "vel_face_ym", 1);
        storage.vel_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_zp", "vel_face_zp", 1);
        storage.vel_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_face_zm", "vel_face_zm", 1);

        storage.press_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_xp", "press_face_xp", 1);
        storage.press_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_xm", "press_face_xm", 1);
        storage.press_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_yp", "press_face_yp", 1);
        storage.press_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_ym", "press_face_ym", 1);
        storage.press_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_zp", "press_face_zp", 1);
        storage.press_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "press_face_zm", "press_face_zm", 1);
    }

    if (solver_config.is_dust_on()) {
        u32 ndust = solver_config.dust_config.ndust;

        storage.rho_dust_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_xp", "rho_dust_face_xp", ndust);
        storage.rho_dust_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_xm", "rho_dust_face_xm", ndust);
        storage.rho_dust_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_yp", "rho_dust_face_yp", ndust);
        storage.rho_dust_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_ym", "rho_dust_face_ym", ndust);
        storage.rho_dust_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_zp", "rho_dust_face_zp", ndust);
        storage.rho_dust_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>>>(
                "rho_dust_face_zm", "rho_dust_face_zm", ndust);

        storage.vel_dust_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_xp", "vel_dust_face_xp", ndust);
        storage.vel_dust_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_xm", "vel_dust_face_xm", ndust);
        storage.vel_dust_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_yp", "vel_dust_face_yp", ndust);
        storage.vel_dust_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_ym", "vel_dust_face_ym", ndust);
        storage.vel_dust_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_zp", "vel_dust_face_zp", ndust);
        storage.vel_dust_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>>>(
                "vel_dust_face_zm", "vel_dust_face_zm", ndust);
    }

    if (solver_config.should_compute_rho_mean()) {
        storage.cell_mass = std::make_shared<shamrock::solvergraph::Field<Tscal>>(
            AMRBlock::block_size, "cell_mass", "m");
        storage.rho_mean
            = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("rho_mean", "< \\rho >");
        storage.simulation_volume = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>(
            "simulation_volume", "V_{\\rm sim}");
    }

    storage.dt_over2
        = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("dt_half", "dt_{half}");

    {

        storage.flux_rho_face_xm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_xm", "flux_rho_face_xm", 1);

        storage.flux_rho_face_xp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_xp", "flux_rho_face_xp", 1);

        storage.flux_rho_face_ym = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_ym", "flux_rho_face_ym", 1);

        storage.flux_rho_face_yp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_yp", "flux_rho_face_yp", 1);

        storage.flux_rho_face_zm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_zm", "flux_rho_face_zm", 1);

        storage.flux_rho_face_zp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rho_face_zp", "flux_rho_face_zp", 1);

        storage.flux_rhov_face_xm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_xm", "flux_rhov_face_xm", 1);

        storage.flux_rhov_face_xp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_xp", "flux_rhov_face_xp", 1);

        storage.flux_rhov_face_ym = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_ym", "flux_rhov_face_ym", 1);

        storage.flux_rhov_face_yp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_yp", "flux_rhov_face_yp", 1);

        storage.flux_rhov_face_zm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_zm", "flux_rhov_face_zm", 1);

        storage.flux_rhov_face_zp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
            "flux_rhov_face_zp", "flux_rhov_face_zp", 1);

        storage.flux_rhoe_face_xm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_xm", "flux_rhoe_face_xm", 1);

        storage.flux_rhoe_face_xp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_xp", "flux_rhoe_face_xp", 1);

        storage.flux_rhoe_face_ym = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_ym", "flux_rhoe_face_ym", 1);

        storage.flux_rhoe_face_yp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_yp", "flux_rhoe_face_yp", 1);

        storage.flux_rhoe_face_zm = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_zm", "flux_rhoe_face_zm", 1);

        storage.flux_rhoe_face_zp = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
            "flux_rhoe_face_zp", "flux_rhoe_face_zp", 1);
    }

    if (solver_config.is_dust_on()) {
        u32 ndust = solver_config.dust_config.ndust;

        storage.flux_rho_dust_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_xm", "flux_rho_dust_face_xm", ndust);

        storage.flux_rho_dust_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_xp", "flux_rho_dust_face_xp", ndust);

        storage.flux_rho_dust_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_ym", "flux_rho_dust_face_ym", ndust);

        storage.flux_rho_dust_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_yp", "flux_rho_dust_face_yp", ndust);

        storage.flux_rho_dust_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_zm", "flux_rho_dust_face_zm", ndust);

        storage.flux_rho_dust_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tscal>>(
                "flux_rho_dust_face_zp", "flux_rho_dust_face_zp", ndust);

        storage.flux_rhov_dust_face_xm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_xm", "flux_rhov_dust_face_xm", ndust);

        storage.flux_rhov_dust_face_xp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_xp", "flux_rhov_dust_face_xp", ndust);

        storage.flux_rhov_dust_face_ym
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_ym", "flux_rhov_dust_face_ym", ndust);

        storage.flux_rhov_dust_face_yp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_yp", "flux_rhov_dust_face_yp", ndust);

        storage.flux_rhov_dust_face_zm
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_zm", "flux_rhov_dust_face_zm", ndust);

        storage.flux_rhov_dust_face_zp
            = std::make_shared<solvergraph::NeighGrapkLinkFieldEdge<Tvec>>(
                "flux_rhov_dust_face_zp", "flux_rhov_dust_face_zp", ndust);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Nodes
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<std::shared_ptr<shamrock::solvergraph::INode>> solver_sequence;

    { // attach fields with ghosts

        { // set element counts
            shamrock::solvergraph::ExtractCounts extract_counts_node
                = shamrock::solvergraph::ExtractCounts();
            extract_counts_node.set_edges(storage.source_patches, storage.block_counts);
            solver_sequence.push_back(
                std::make_shared<decltype(extract_counts_node)>(std::move(extract_counts_node)));
        }

        { // set element counts
            shamrock::solvergraph::ExtractCounts extract_counts_node
                = shamrock::solvergraph::ExtractCounts();
            extract_counts_node.set_edges(
                storage.merged_patchdata_ghost, storage.block_counts_with_ghost);
            solver_sequence.push_back(
                std::make_shared<decltype(extract_counts_node)>(std::move(extract_counts_node)));
        }

        { // Attach spans to block coords
            shamrock::solvergraph::GetFieldRefFromLayer<TgridVec> attach_block_min
                = shamrock::solvergraph::GetFieldRefFromLayer<TgridVec>(0);
            attach_block_min.set_edges(storage.merged_patchdata_ghost, storage.refs_block_min);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_block_min)>(std::move(attach_block_min)));

            shamrock::solvergraph::GetFieldRefFromLayer<TgridVec> attach_block_max
                = shamrock::solvergraph::GetFieldRefFromLayer<TgridVec>(1);
            attach_block_max.set_edges(storage.merged_patchdata_ghost, storage.refs_block_max);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_block_max)>(std::move(attach_block_max)));
        }

        { // attach spans to gas field with ghosts
            shamrock::solvergraph::GetFieldRefFromLayer<Tscal> attach_rho
                = shamrock::solvergraph::GetFieldRefFromLayer<Tscal>(storage.ghost_layout, "rho");
            attach_rho.set_edges(storage.merged_patchdata_ghost, storage.refs_rho);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_rho)>(std::move(attach_rho)));

            shamrock::solvergraph::GetFieldRefFromLayer<Tvec> attach_rhov
                = shamrock::solvergraph::GetFieldRefFromLayer<Tvec>(storage.ghost_layout, "rhovel");
            attach_rhov.set_edges(storage.merged_patchdata_ghost, storage.refs_rhov);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_rhov)>(std::move(attach_rhov)));

            shamrock::solvergraph::GetFieldRefFromLayer<Tscal> attach_rhoe
                = shamrock::solvergraph::GetFieldRefFromLayer<Tscal>(
                    storage.ghost_layout, "rhoetot");
            attach_rhoe.set_edges(storage.merged_patchdata_ghost, storage.refs_rhoe);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_rhoe)>(std::move(attach_rhoe)));
        }

        if (solver_config.is_dust_on()) { // attach spans to dust field with ghosts
            shamrock::solvergraph::GetFieldRefFromLayer<Tscal> attach_rho_dust
                = shamrock::solvergraph::GetFieldRefFromLayer<Tscal>(
                    storage.ghost_layout, "rho_dust");
            attach_rho_dust.set_edges(storage.merged_patchdata_ghost, storage.refs_rho_dust);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_rho_dust)>(std::move(attach_rho_dust)));

            shamrock::solvergraph::GetFieldRefFromLayer<Tvec> attach_rhov_dust
                = shamrock::solvergraph::GetFieldRefFromLayer<Tvec>(
                    storage.ghost_layout, "rhovel_dust");
            attach_rhov_dust.set_edges(storage.merged_patchdata_ghost, storage.refs_rhov_dust);
            solver_sequence.push_back(
                std::make_shared<decltype(attach_rhov_dust)>(std::move(attach_rhov_dust)));
        }
    }

    { // build trees

        modules::NodeBuildTrees<u_morton, TgridVec> node{};
        node.set_edges(
            storage.block_counts_with_ghost,
            storage.refs_block_min,
            storage.refs_block_max,
            storage.trees);

        solver_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    { // build neigh tables
        std::vector<std::shared_ptr<shamrock::solvergraph::INode>> neigh_table_sequence;

        modules::FindBlockNeigh<Tvec, TgridVec, u_morton> node1;
        node1.set_edges(
            storage.block_counts_with_ghost,
            storage.refs_block_min,
            storage.refs_block_max,
            storage.trees,
            storage.block_graph_edge);
        node1.evaluate();

        modules::BlockNeighToCellNeigh<Tvec, TgridVec, u_morton> node2(Config::NsideBlockPow);
        node2.set_edges(
            storage.block_counts_with_ghost,
            storage.refs_block_min,
            storage.refs_block_max,
            storage.block_graph_edge,
            storage.cell_graph_edge);
        node2.evaluate();

        neigh_table_sequence.push_back(std::make_shared<decltype(node1)>(std::move(node1)));
        get_optional_free_mem(storage.trees, neigh_table_sequence);
        neigh_table_sequence.push_back(std::make_shared<decltype(node2)>(std::move(node2)));
        get_optional_free_mem(storage.block_graph_edge, neigh_table_sequence);

        shamrock::solvergraph::OperationSequence seq(
            "Compute neigh table", std::move(neigh_table_sequence));
        solver_sequence.push_back(std::make_shared<decltype(seq)>(std::move(seq)));
    }

    { // Compute cell infos

        modules::NodeComputeCellAABB<Tvec, TgridVec> node{
            AMRBlock::Nside, solver_config.grid_coord_to_pos_fact};

        node.set_edges(
            storage.block_counts_with_ghost,
            storage.refs_block_min,
            storage.refs_block_max,
            storage.block_cell_sizes,
            storage.cell0block_aabb_lower);
        solver_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    if (solver_config.should_compute_rho_mean()) {
        modules::NodeComputeMass<Tvec, TgridVec> node{AMRBlock::block_size};
        node.set_edges(
            storage.block_counts, storage.block_cell_sizes, storage.refs_rho, storage.cell_mass);
        solver_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));

        modules::NodeComputeSumOverV<Tscal> node2{AMRBlock::block_size};
        node2.set_edges(
            storage.block_counts, storage.cell_mass, storage.simulation_volume, storage.rho_mean);
        solver_sequence.push_back(std::make_shared<decltype(node2)>(std::move(node2)));
    }

    { // Build ConsToPrim node
        std::vector<std::shared_ptr<shamrock::solvergraph::INode>> const_to_prim_sequence;

        {
            modules::NodeConsToPrimGas<Tvec> node{AMRBlock::block_size, solver_config.eos_gamma};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.refs_rho,
                storage.refs_rhov,
                storage.refs_rhoe,
                storage.vel,
                storage.press);

            const_to_prim_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        if (solver_config.is_dust_on()) {
            u32 ndust = solver_config.dust_config.ndust;
            modules::NodeConsToPrimDust<Tvec> node{AMRBlock::block_size, ndust};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.refs_rho_dust,
                storage.refs_rhov_dust,
                storage.vel_dust);

            const_to_prim_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        shamrock::solvergraph::OperationSequence seq(
            "Cons to Prim", std::move(const_to_prim_sequence));
        solver_sequence.push_back(std::make_shared<decltype(seq)>(std::move(seq)));
    }

    { // Build slope limited gradients

        std::vector<std::shared_ptr<shamrock::solvergraph::INode>> grad_sequence;

        {
            modules::SlopeLimitedScalarGradient<Tvec, TgridVec> node{
                AMRBlock::block_size, 1, solver_config.slope_config};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.refs_rho,
                storage.grad_rho);
            grad_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        {
            modules::SlopeLimitedVectorGradient<Tvec, TgridVec> node{
                AMRBlock::block_size, 1, solver_config.slope_config};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.vel,
                storage.dx_v,
                storage.dy_v,
                storage.dz_v);
            grad_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }
        {
            modules::SlopeLimitedScalarGradient<Tvec, TgridVec> node{
                AMRBlock::block_size, 1, solver_config.slope_config};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.press,
                storage.grad_P);
            grad_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        if (solver_config.is_dust_on()) {
            u32 ndust = solver_config.dust_config.ndust;
            modules::SlopeLimitedScalarGradient<Tvec, TgridVec> node{
                AMRBlock::block_size, ndust, solver_config.slope_config};
            node.set_edges(
                storage.block_counts_with_ghost,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.refs_rho_dust,
                storage.grad_rho_dust);
            grad_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));

            modules::SlopeLimitedVectorGradient<Tvec, TgridVec> node2{
                AMRBlock::block_size, ndust, solver_config.slope_config};
            node2.set_edges(
                storage.block_counts_with_ghost,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.vel_dust,
                storage.dx_v_dust,
                storage.dy_v_dust,
                storage.dz_v_dust);
            grad_sequence.push_back(std::make_shared<decltype(node2)>(std::move(node2)));
        }

        shamrock::solvergraph::OperationSequence seq(
            "Slope limited gradients", std::move(grad_sequence));
        solver_sequence.push_back(std::make_shared<decltype(seq)>(std::move(seq)));
    }

    { // interpolate to face
        std::vector<std::shared_ptr<shamrock::solvergraph::INode>> interp_sequence;
        {
            modules::InterpolateToFaceRho<Tvec, TgridVec> node{AMRBlock::block_size};
            node.set_edges(
                storage.dt_over2,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.cell0block_aabb_lower,
                storage.refs_rho,
                storage.grad_rho,
                storage.vel,
                storage.dx_v,
                storage.dy_v,
                storage.dz_v,
                storage.rho_face_xp,
                storage.rho_face_xm,
                storage.rho_face_yp,
                storage.rho_face_ym,
                storage.rho_face_zp,
                storage.rho_face_zm);
            interp_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        {
            modules::InterpolateToFaceVel<Tvec, TgridVec> node{AMRBlock::block_size};
            node.set_edges(
                storage.dt_over2,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.cell0block_aabb_lower,
                storage.refs_rho,
                storage.grad_P,
                storage.vel,
                storage.dx_v,
                storage.dy_v,
                storage.dz_v,
                storage.vel_face_xp,
                storage.vel_face_xm,
                storage.vel_face_yp,
                storage.vel_face_ym,
                storage.vel_face_zp,
                storage.vel_face_zm);
            interp_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        {
            modules::InterpolateToFacePress<Tvec, TgridVec> node{
                AMRBlock::block_size, solver_config.eos_gamma};
            node.set_edges(
                storage.dt_over2,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.cell0block_aabb_lower,
                storage.press,
                storage.grad_P,
                storage.vel,
                storage.dx_v,
                storage.dy_v,
                storage.dz_v,
                storage.press_face_xp,
                storage.press_face_xm,
                storage.press_face_yp,
                storage.press_face_ym,
                storage.press_face_zp,
                storage.press_face_zm);
            interp_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        if (solver_config.is_dust_on()) {
            u32 ndust = solver_config.dust_config.ndust;
            modules::InterpolateToFaceRhoDust<Tvec, TgridVec> node{AMRBlock::block_size, ndust};
            node.set_edges(
                storage.dt_over2,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.cell0block_aabb_lower,
                storage.refs_rho_dust,
                storage.grad_rho_dust,
                storage.vel_dust,
                storage.dx_v_dust,
                storage.dy_v_dust,
                storage.dz_v_dust,
                storage.rho_dust_face_xp,
                storage.rho_dust_face_xm,
                storage.rho_dust_face_yp,
                storage.rho_dust_face_ym,
                storage.rho_dust_face_zp,
                storage.rho_dust_face_zm);
            interp_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        if (solver_config.is_dust_on()) {
            u32 ndust = solver_config.dust_config.ndust;
            modules::InterpolateToFaceVelDust<Tvec, TgridVec> node{AMRBlock::block_size, ndust};
            node.set_edges(
                storage.dt_over2,
                storage.cell_graph_edge,
                storage.block_cell_sizes,
                storage.cell0block_aabb_lower,
                storage.refs_rho_dust,
                storage.vel_dust,
                storage.dx_v_dust,
                storage.dy_v_dust,
                storage.dz_v_dust,
                storage.vel_dust_face_xp,
                storage.vel_dust_face_xm,
                storage.vel_dust_face_yp,
                storage.vel_dust_face_ym,
                storage.vel_dust_face_zp,
                storage.vel_dust_face_zm);
            interp_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        }

        shamrock::solvergraph::OperationSequence seq(
            "Interpolate to face", std::move(interp_sequence));
        solver_sequence.push_back(std::make_shared<decltype(seq)>(std::move(seq)));
    }

    { // flux

        std::vector<std::shared_ptr<shamrock::solvergraph::INode>> flux_sequence;

        if (solver_config.riemman_config == Rusanov) {
            modules::NodeComputeFluxGasMode<Tvec, TgridVec, modules::RiemannSolverMode::Rusanov>
                node(
                    "Gas flux compute",
                    solver_config.eos_gamma,
                    storage.cell_graph_edge,
                    storage.rho_face_xp,
                    storage.rho_face_xm,
                    storage.rho_face_yp,
                    storage.rho_face_ym,
                    storage.rho_face_zp,
                    storage.rho_face_zm,
                    storage.vel_face_xp,
                    storage.vel_face_xm,
                    storage.vel_face_yp,
                    storage.vel_face_ym,
                    storage.vel_face_zp,
                    storage.vel_face_zm,
                    storage.press_face_xp,
                    storage.press_face_xm,
                    storage.press_face_yp,
                    storage.press_face_ym,
                    storage.press_face_zp,
                    storage.press_face_zm,
                    storage.flux_rho_face_xp,
                    storage.flux_rho_face_xm,
                    storage.flux_rho_face_yp,
                    storage.flux_rho_face_ym,
                    storage.flux_rho_face_zp,
                    storage.flux_rho_face_zm,
                    storage.flux_rhov_face_xp,
                    storage.flux_rhov_face_xm,
                    storage.flux_rhov_face_yp,
                    storage.flux_rhov_face_ym,
                    storage.flux_rhov_face_zp,
                    storage.flux_rhov_face_zm,
                    storage.flux_rhoe_face_xp,
                    storage.flux_rhoe_face_xm,
                    storage.flux_rhoe_face_yp,
                    storage.flux_rhoe_face_ym,
                    storage.flux_rhoe_face_zp,
                    storage.flux_rhoe_face_zm);
            flux_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        } else if (solver_config.riemman_config == HLL) {
            modules::NodeComputeFluxGasMode<Tvec, TgridVec, modules::RiemannSolverMode::HLL> node(
                "Gas flux compute",
                solver_config.eos_gamma,
                storage.cell_graph_edge,
                storage.rho_face_xp,
                storage.rho_face_xm,
                storage.rho_face_yp,
                storage.rho_face_ym,
                storage.rho_face_zp,
                storage.rho_face_zm,
                storage.vel_face_xp,
                storage.vel_face_xm,
                storage.vel_face_yp,
                storage.vel_face_ym,
                storage.vel_face_zp,
                storage.vel_face_zm,
                storage.press_face_xp,
                storage.press_face_xm,
                storage.press_face_yp,
                storage.press_face_ym,
                storage.press_face_zp,
                storage.press_face_zm,
                storage.flux_rho_face_xp,
                storage.flux_rho_face_xm,
                storage.flux_rho_face_yp,
                storage.flux_rho_face_ym,
                storage.flux_rho_face_zp,
                storage.flux_rho_face_zm,
                storage.flux_rhov_face_xp,
                storage.flux_rhov_face_xm,
                storage.flux_rhov_face_yp,
                storage.flux_rhov_face_ym,
                storage.flux_rhov_face_zp,
                storage.flux_rhov_face_zm,
                storage.flux_rhoe_face_xp,
                storage.flux_rhoe_face_xm,
                storage.flux_rhoe_face_yp,
                storage.flux_rhoe_face_ym,
                storage.flux_rhoe_face_zp,
                storage.flux_rhoe_face_zm);
            flux_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        } else if (solver_config.riemman_config == HLLC) {
            modules::NodeComputeFluxGasMode<Tvec, TgridVec, modules::RiemannSolverMode::HLLC> node(
                "Gas flux compute",
                solver_config.eos_gamma,
                storage.cell_graph_edge,
                storage.rho_face_xp,
                storage.rho_face_xm,
                storage.rho_face_yp,
                storage.rho_face_ym,
                storage.rho_face_zp,
                storage.rho_face_zm,
                storage.vel_face_xp,
                storage.vel_face_xm,
                storage.vel_face_yp,
                storage.vel_face_ym,
                storage.vel_face_zp,
                storage.vel_face_zm,
                storage.press_face_xp,
                storage.press_face_xm,
                storage.press_face_yp,
                storage.press_face_ym,
                storage.press_face_zp,
                storage.press_face_zm,
                storage.flux_rho_face_xp,
                storage.flux_rho_face_xm,
                storage.flux_rho_face_yp,
                storage.flux_rho_face_ym,
                storage.flux_rho_face_zp,
                storage.flux_rho_face_zm,
                storage.flux_rhov_face_xp,
                storage.flux_rhov_face_xm,
                storage.flux_rhov_face_yp,
                storage.flux_rhov_face_ym,
                storage.flux_rhov_face_zp,
                storage.flux_rhov_face_zm,
                storage.flux_rhoe_face_xp,
                storage.flux_rhoe_face_xm,
                storage.flux_rhoe_face_yp,
                storage.flux_rhoe_face_ym,
                storage.flux_rhoe_face_zp,
                storage.flux_rhoe_face_zm);
            flux_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
        } else {
            shambase::throw_unimplemented("unknown flux mode");
        }

        if (solver_config.is_dust_on()) {
            u32 ndust = solver_config.dust_config.ndust;
            if (solver_config.dust_config.dust_riemann_config == DHLL) {
                modules::
                    NodeComputeFluxDustMode<Tvec, TgridVec, modules::DustRiemannSolverMode::DHLL>
                        node(
                            "Dust flux compute",
                            ndust,
                            storage.cell_graph_edge,
                            storage.rho_dust_face_xp,
                            storage.rho_dust_face_xm,
                            storage.rho_dust_face_yp,
                            storage.rho_dust_face_ym,
                            storage.rho_dust_face_zp,
                            storage.rho_dust_face_zm,
                            storage.vel_dust_face_xp,
                            storage.vel_dust_face_xm,
                            storage.vel_dust_face_yp,
                            storage.vel_dust_face_ym,
                            storage.vel_dust_face_zp,
                            storage.vel_dust_face_zm,
                            storage.flux_rho_dust_face_xp,
                            storage.flux_rho_dust_face_xm,
                            storage.flux_rho_dust_face_yp,
                            storage.flux_rho_dust_face_ym,
                            storage.flux_rho_dust_face_zp,
                            storage.flux_rho_dust_face_zm,
                            storage.flux_rhov_dust_face_xp,
                            storage.flux_rhov_dust_face_xm,
                            storage.flux_rhov_dust_face_yp,
                            storage.flux_rhov_dust_face_ym,
                            storage.flux_rhov_dust_face_zp,
                            storage.flux_rhov_dust_face_zm);
                flux_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
            } else if (solver_config.dust_config.dust_riemann_config == HB) {
                modules::NodeComputeFluxDustMode<Tvec, TgridVec, modules::DustRiemannSolverMode::HB>
                    node(
                        "Dust flux compute",
                        ndust,
                        storage.cell_graph_edge,
                        storage.rho_dust_face_xp,
                        storage.rho_dust_face_xm,
                        storage.rho_dust_face_yp,
                        storage.rho_dust_face_ym,
                        storage.rho_dust_face_zp,
                        storage.rho_dust_face_zm,
                        storage.vel_dust_face_xp,
                        storage.vel_dust_face_xm,
                        storage.vel_dust_face_yp,
                        storage.vel_dust_face_ym,
                        storage.vel_dust_face_zp,
                        storage.vel_dust_face_zm,
                        storage.flux_rho_dust_face_xp,
                        storage.flux_rho_dust_face_xm,
                        storage.flux_rho_dust_face_yp,
                        storage.flux_rho_dust_face_ym,
                        storage.flux_rho_dust_face_zp,
                        storage.flux_rho_dust_face_zm,
                        storage.flux_rhov_dust_face_xp,
                        storage.flux_rhov_dust_face_xm,
                        storage.flux_rhov_dust_face_yp,
                        storage.flux_rhov_dust_face_ym,
                        storage.flux_rhov_dust_face_zp,
                        storage.flux_rhov_dust_face_zm);
                flux_sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
            } else {
                shambase::throw_unimplemented("unknown flux mode");
            }
        }

        shamrock::solvergraph::OperationSequence seq("Compute fluxes", std::move(flux_sequence));
        solver_sequence.push_back(std::make_shared<decltype(seq)>(std::move(seq)));
    }

    shamrock::solvergraph::OperationSequence seq("Solver", std::move(solver_sequence));
    storage.solver_sequence = std::make_shared<decltype(seq)>(std::move(seq));

    if (false) {
        logger::raw_ln(" -- tex:\n" + shambase::get_check_ref(storage.solver_sequence).get_tex());
        logger::raw_ln(
            " -- dot:\n" + shambase::get_check_ref(storage.solver_sequence).get_dot_graph());
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::evolve_once() {

    StackEntry stack_loc{};

    sham::MemPerfInfos mem_perf_infos_start = sham::details::get_mem_perf_info();
    f64 mpi_timer_start                     = shamcomm::mpi::get_timer("total");

    Tscal t_current = solver_config.get_time();
    Tscal dt_input  = solver_config.get_dt();

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("amr::Godunov", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

    if (solver_config.face_half_time_interpolation) {
        shambase::get_check_ref(storage.dt_over2).value = dt_input / 2.0;
    }

    shambase::Timer tstep;
    tstep.start();

    // Scheduler step
    auto update_load_val = [&]() {
        shamlog_debug_ln("ComputeLoadBalanceValue", "update load balancing");
        scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
            return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
        });
    };
    update_load_val();
    scheduler().scheduler_step(true, true);
    update_load_val();
    scheduler().scheduler_step(false, false);

    if (solver_config.should_compute_rho_mean()) {
        auto [bmin, bmax] = scheduler().template get_box_volume<TgridVec>();
        Tscal dxfact      = solver_config.grid_coord_to_pos_fact;
        Tvec dV           = (bmax - bmin).template convert<Tscal>() * dxfact;
        Tscal Vsim        = dV.x() * dV.y() * dV.z();
        shambase::get_check_ref(storage.simulation_volume).value = Vsim;
    }

    // give to the solvergraph the patch rank owners
    storage.patch_rank_owner->values = {};
    scheduler().for_each_global_patch([&](const shamrock::patch::Patch p) {
        storage.patch_rank_owner->values.add_obj(
            p.id_patch, scheduler().get_patch_rank_owner(p.id_patch));
    });

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch &p, shamrock::patch::PatchDataLayer &pdat) {
            storage.source_patches->patchdatas.add_obj(p.id_patch, std::ref(pdat));
        });

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    // ghost zone exchange
    modules::GhostZones gz(context, solver_config, storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();

    // compute prim variable
    {
        // logger::raw_ln(" -- tex:\n" +
        // shambase::get_check_ref(storage.solver_sequence).get_tex());
        // logger::raw_ln(
        //   " -- dot:\n" + shambase::get_check_ref(storage.solver_sequence).get_dot_graph());
        shambase::get_check_ref(storage.solver_sequence).evaluate();
    }

    // flux
    // modules::ComputeFlux flux_compute(context, solver_config, storage);
    // flux_compute.compute_flux();
    // if (solver_config.is_dust_on()) {
    //    flux_compute.compute_flux_dust();
    //}

    // compute dt fields
    modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
    dt_compute.compute_dt_fields();
    if (solver_config.is_dust_on()) {
        dt_compute.compute_dt_dust_fields();
    }

    // RK2 + flux lim
    if (solver_config.drag_config.drag_solver_config == DragSolverMode::NoDrag) {
        modules::TimeIntegrator dt_integ(context, solver_config, storage);
        dt_integ.forward_euler(dt_input);
    } else if (solver_config.drag_config.drag_solver_config == DragSolverMode::IRK1) {
        modules::DragIntegrator drag_integ(context, solver_config, storage);
        drag_integ.involve_with_no_src(dt_input);
        drag_integ.enable_irk1_drag_integrator(dt_input);
    } else if (solver_config.drag_config.drag_solver_config == DragSolverMode::EXPO) {
        modules::DragIntegrator drag_integ(context, solver_config, storage);
        drag_integ.involve_with_no_src(dt_input);
        drag_integ.enable_expo_drag_integrator(dt_input);
    } else {
        shambase::throw_unimplemented();
    }

    modules::AMRGridRefinementHandler refinement(context, solver_config, storage);
    refinement.update_refinement();

    modules::ComputeCFL cfl_compute(context, solver_config, storage);
    f64 new_dt = cfl_compute.compute_cfl();

    // if new physics like dust is added then use the smallest dt
    if (solver_config.is_dust_on())
        new_dt = std::min(new_dt, cfl_compute.compute_dust_cfl());

    solver_config.set_next_dt(new_dt);
    solver_config.set_time(t_current + dt_input);

    storage.dtrho.reset();
    storage.dtrhov.reset();
    storage.dtrhoe.reset();

    if (solver_config.is_dust_on()) {
        storage.dtrho_dust.reset();
        storage.dtrhov_dust.reset();
    }

    if (solver_config.drag_config.drag_solver_config != DragSolverMode::NoDrag) {
        storage.rho_next_no_drag.reset();
        storage.rhov_next_no_drag.reset();
        storage.rhoe_next_no_drag.reset();
        storage.rho_d_next_no_drag.reset();
        storage.rhov_d_next_no_drag.reset();
    }

    storage.merge_patch_bounds.reset();

    storage.ghost_zone_infos.reset();

    storage.serial_patch_tree.reset();

    storage.source_patches->free_alloc();

    tstep.end();

    sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

    f64 delta_mpi_timer = shamcomm::mpi::get_timer("total") - mpi_timer_start;
    f64 t_dev_alloc
        = (mem_perf_infos_end.time_alloc_device - mem_perf_infos_start.time_alloc_device)
          + (mem_perf_infos_end.time_free_device - mem_perf_infos_start.time_free_device);

    u64 rank_count = scheduler().get_rank_count() * AMRBlock::block_size;
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

    std::string log_step = report_perf_timestep(
        rate,
        rank_count,
        tstep.elasped_sec(),
        delta_mpi_timer,
        t_dev_alloc,
        mem_perf_infos_end.max_allocated_byte_device);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::RAMSES", log_step);
        logger::info_ln(
            "amr::RAMSES",
            "estimated rate :",
            dt_input * (3600 / tstep.elasped_sec()),
            "(tsim/hr)");
    }

    storage.timings_details.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::do_debug_vtk_dump(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    u32 block_size = Solver::AMRBlock::block_size;

    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
    std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

    sycl::buffer<Tvec> pos_min_cell(num_obj * block_size);
    sycl::buffer<Tvec> pos_max_cell(num_obj * block_size);

    shamsys::instance::get_compute_queue().submit([&, block_size](sycl::handler &cgh) {
        sycl::accessor acc_p1{shambase::get_check_ref(pos1), cgh, sycl::read_only};
        sycl::accessor acc_p2{shambase::get_check_ref(pos2), cgh, sycl::read_only};
        sycl::accessor cell_min{pos_min_cell, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor cell_max{pos_max_cell, cgh, sycl::write_only, sycl::no_init};

        using Block = typename Solver::AMRBlock;

        shambase::parallel_for(cgh, num_obj, "rescale cells", [=](u64 id_a) {
            Tvec block_min = acc_p1[id_a].template convert<Tscal>();
            Tvec block_max = acc_p2[id_a].template convert<Tscal>();

            Tvec delta_cell = (block_max - block_min) / Block::side_size;
#pragma unroll
            for (u32 ix = 0; ix < Block::side_size; ix++) {
#pragma unroll
                for (u32 iy = 0; iy < Block::side_size; iy++) {
#pragma unroll
                    for (u32 iz = 0; iz < Block::side_size; iz++) {
                        u32 i                           = Block::get_index({ix, iy, iz});
                        Tvec delta_val                  = delta_cell * Tvec{ix, iy, iz};
                        cell_min[id_a * block_size + i] = block_min + delta_val;
                        cell_max[id_a * block_size + i] = block_min + (delta_cell) + delta_val;
                    }
                }
            }
        });
    });

    writer.write_voxel_cells(pos_min_cell, pos_max_cell, num_obj * block_size);

    writer.add_cell_data_section();
    writer.add_field_data_section(6);

    std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", fields_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(3);
    writer.write_field("rhovel", fields_vel, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(4);
    writer.write_field("rhoetot", fields_eint, num_obj * block_size);
    /*
        std::unique_ptr<sycl::buffer<Tvec>> grad_rho
            = storage.grad_rho.get().rankgather_computefield(sched);
        writer.write_field("grad_rho", grad_rho, num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tvec>> dx_v =
       storage.dx_v.get().rankgather_computefield(sched); writer.write_field("dx_v", dx_v,
       num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tvec>> dy_v =
       storage.dy_v.get().rankgather_computefield(sched); writer.write_field("dy_v", dy_v,
       num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tvec>> dz_v =
       storage.dz_v.get().rankgather_computefield(sched); writer.write_field("dz_v", dz_v,
       num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tvec>> grad_P
            = storage.grad_P.get().rankgather_computefield(sched);
        writer.write_field("grad_P", grad_P, num_obj * block_size);
    */
    std::unique_ptr<sycl::buffer<Tscal>> dtrho = storage.dtrho.get().rankgather_computefield(sched);
    writer.write_field("dtrho", dtrho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dtrhov
        = storage.dtrhov.get().rankgather_computefield(sched);
    writer.write_field("dtrhov", dtrhov, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrhoe
        = storage.dtrhoe.get().rankgather_computefield(sched);
    writer.write_field("dtrhoe", dtrhoe, num_obj * block_size);
}

template class shammodels::basegodunov::Solver<f64_3, i64_3>;
