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
 * @file NodeAYPX.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node computing the in place update y <- alpha * y + x on fields
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<T>, spans_x)                                            \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, alpha)                                           \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<T>, spans_y)

namespace shamrock::solvergraph {

    /**
     * @brief Compute `y <- alpha * y + x` on every value of every patch.
     *
     * `sizes` holds the object counts per patch, each object holding `nvar` values (e.g. the cells
     * of an AMR block). `spans_y` is
     * updated in place, so it must already be allocated with the right sizes.
     */
    template<class T>
    class NodeAYPX : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<T>;

        u32 nvar;

        public:
        NodeAYPX(u32 nvar) : nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeAYPX"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shamrock::solvergraph

#undef NODE_EDGES
