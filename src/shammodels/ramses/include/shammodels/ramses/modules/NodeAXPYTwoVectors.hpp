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
 * @file NodeAXPYTwoVectors.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

#define NODE_AXPY_EDGES(X_RO, X_RW)                                                                \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes_no_gz)                                         \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, spans_x)                                            \
    X_RO(shamrock::solvergraph::ScalarEdge<f64>, alpha)                                            \
    /* outputs*/                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<T>, spans_y)

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeAXPYTwoVectors : public shamrock::solvergraph::INode {

        u32 block_size;

        public:
        NodeAXPYTwoVectors(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_AXPY_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeAXPYTwoVectors"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_AXPY_EDGES
