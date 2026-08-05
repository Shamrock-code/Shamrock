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
 * @file NodeSpMVPoisson3D.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Implementation of SpMV [A*p], A the 3D SPD sparse Poisson matrix.
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

#define NODE_SPMV_EDGES(X_RO, X_RW)                                                                \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::Field<Tscal>, spans_block_cell_sizes)                              \
    X_RO(shamrock::solvergraph::Field<Tscal>, spans_in)                                            \
    /* outputs*/                                                                                   \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_out)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeSpMVPoisson3D : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;
        u32 block_size;

        public:
        NodeSpMVPoisson3D(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_SPMV_EDGES)

        void _impl_evaluate_internal();
        inline virtual std::string _impl_get_label() const { return "SpMVPoisson3D"; };
        virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_SPMV_EDGES
