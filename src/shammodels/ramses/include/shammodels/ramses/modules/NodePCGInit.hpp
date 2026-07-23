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
 * @file NodePCGInit.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>

#define NODE_PCGINIT_EDGES(X_RO, X_RW)                                                             \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mean_rho)                                       \
    /* outputs*/                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_res)                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_pres)                                 \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_phi_p)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class PCGInit : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        u32 block_size;
        Tscal fourPiG;

        public:
        PCGInit(u32 block_size, Tscal fourPiG) : block_size(block_size), fourPiG(fourPiG) {}

        EXPAND_NODE_EDGES(NODE_PCGINIT_EDGES)

        void _impl_evaluate_internal();
        inline virtual std::string _impl_get_label() const { return "PCGInit"; };
        virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_PCGINIT_EDGES
