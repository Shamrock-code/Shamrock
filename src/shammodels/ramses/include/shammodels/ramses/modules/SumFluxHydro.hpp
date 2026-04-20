// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SumFluxHydro.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Sum the fluxes into the time derivative fields for Hydro
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

#define NODE_SUM_FLUX_HYDRO(X_RO, X_RW)                                                            \
    /* ------------------- inputs ------------------- */                                           \
    /* number of blocks     */                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, block_counts)                                        \
                                                                                                   \
    /* cell neighbor graph */                                                                      \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
                                                                                                   \
    /* block cell sizes */                                                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    /* block low corner position */                                                                \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_cell0block_aabb_lower)                     \
                                                                                                   \
    /* fluxes */                                                                                   \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_xp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_xm)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_yp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_ym)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_zp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rho_face_zm)                            \
                                                                                                   \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_xp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_xm)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_yp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_ym)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_zp)                            \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tvec>, flux_rhov_face_zm)                            \
                                                                                                   \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_xp)                           \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_xm)                           \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_yp)                           \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_ym)                           \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_zp)                           \
    X_RO(solvergraph::NeighGraphLinkFieldEdge<Tscal>, flux_rhoe_face_zm)                           \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dtrho)  /* time derivative of density*/   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dtrhov)  /* time derivative of momentum*/  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dtrhoe) /* time derivative of energy*/

namespace shammodels::basegodunov::modules {
    template<class T>
    using NGLink = shammodels::basegodunov::modules::NeighGraphLinkField<T>;

    template<class Tvec, class TgridVec>
    class NodeSumFluxHydro : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;

        u32 block_size;
        Tscal dxfact;

        public:
        NodeSumFluxHydro(u32 block_size, Tscal dxfact) : block_size(block_size), dxfact(dxfact) {}

        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        EXPAND_NODE_EDGES(NODE_SUM_FLUX_HYDRO)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SumFluxHydro"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_SUM_FLUX_HYDRO
