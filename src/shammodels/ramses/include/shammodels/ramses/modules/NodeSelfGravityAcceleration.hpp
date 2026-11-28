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
 * @file NodeSelfGravityAcceleration.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Self-gravity acceleration \f$ \mathbf{g} = -\nabla \phi \f$ on the AMR grid.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <string>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi)                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_phi_g)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the gravitational acceleration \f$ \mathbf{g} = -\nabla \phi \f$ of every
     * cell.
     *
     * In each direction the gradient is the mean of the gradients across the two faces, the
     * gradient across a face being the mean over the neighbours of \f$ (\phi_b - \phi_i) / d \f$,
     * with the same distances \f$ d \f$ between the cell centers as in laplacian_7pt
     * (\f$ \Delta x \f$ for a neighbour of the same level, \f$ 3 \Delta x / 2 \f$ for a coarser
     * one and \f$ 3 \Delta x / 4 \f$ for a finer one). On a uniform grid this is the second order
     * centered difference.
     *
     * TODO: (kept from #1942) a face without any neighbour contributes a zero gradient that is
     * still averaged with the other face, which halves the gradient in that direction (only
     * matters for non periodic boundaries).
     *
     * `sizes` holds the block counts per patch, each block holding `block_size` cells.
     * `spans_phi_g` is resized to match `sizes` if needed.
     */
    template<class Tvec, class TgridVec>
    class NodeSelfGravityAcceleration : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;

        public:
        NodeSelfGravityAcceleration(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SelfGravityAcceleration"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
