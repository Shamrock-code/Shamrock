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
 * @file ComputeCellAABB.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, block_min)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, block_max)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, block_cell_sizes)                               \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, cell0block_aabb_lower)

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeComputeCellAABB : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_nside;
        Tscal grid_coord_to_pos_fact;

        public:
        NodeComputeCellAABB(u32 block_nside, Tscal grid_coord_to_pos_fact)
            : block_nside(block_nside), grid_coord_to_pos_fact(grid_coord_to_pos_fact) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeCellAABB"; }

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
