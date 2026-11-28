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
 * @file BiCGSTABInit.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Initialisation step of the BiCGSTAB Poisson solver.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mean_rho)                                       \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_res)                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_res_bis)                              \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_p)                                    \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhs)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Initialisation step of BiCGSTAB: compute the right-hand side \f$ b \f$, the residual
     * \f$ r_0 = b - A \phi_0 \f$, the shadow residual \f$ r'_0 = r_0 / 2 \f$ and the search
     * direction
     * \f$ p_0 = r_0 \f$.
     *
     * The matrix and the right-hand side are the ones of CGInit:
     * \f$ b = -4 \pi G (\rho - \bar{\rho}) \Delta x^3 \f$. `sizes` holds the block counts per
     * patch, each block holding `block_size` cells. The outputs must already be allocated with the
     * right sizes.
     */
    template<class Tvec, class TgridVec>
    class BiCGSTABInit : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        Tscal fourPiG;

        public:
        BiCGSTABInit(u32 block_size, Tscal fourPiG) : block_size(block_size), fourPiG(fourPiG) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "BiCGSTABInit"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
