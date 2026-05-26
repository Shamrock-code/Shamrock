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
 * @file BlockNeighToCellNeigh.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shammodels/ramses/solvegraph/TreeEdge.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldRefs<TgridVec>, spans_block_min)                             \
    X_RO(shamrock::solvergraph::IFieldRefs<TgridVec>, spans_block_max)                             \
    X_RO(OrientedAMRGraph, block_neigh_graph)                                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(OrientedAMRGraph, cell_neigh_graph)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class BlockNeighToCellNeigh : public shamrock::solvergraph::INode {
        using Tscal            = shambase::VecComponent<Tvec>;
        using RTree            = RadixTree<Tmorton, TgridVec>;
        using OrientedAMRGraph = OrientedAMRGraph<Tvec, TgridVec>;

        template<class AMRBlock>
        class AMRLowering;

        u32 block_nside_pow;

        public:
        BlockNeighToCellNeigh(u32 block_nside_pow) : block_nside_pow(block_nside_pow) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "BlockNeighToCellNeigh"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
