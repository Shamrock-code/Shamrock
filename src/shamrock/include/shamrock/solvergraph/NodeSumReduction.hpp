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
 * @file NodeSumReduction.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node computing the global sum of a field over all patches and ranks
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, spans_in)                                           \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IDataEdge<T>, out_scal)

namespace shamrock::solvergraph {

    /**
     * @brief Compute the sum of `spans_in` over every value of every patch on every rank.
     *
     * `sizes` holds the object counts per patch, each object holding `nvar` values (e.g. the cells
     * of an AMR block). The result is
     * deterministic: each patch is reduced with the `shamalgs` primitives, the patches are
     * accumulated in increasing patch id order, then summed across ranks.
     */
    template<class T>
    class NodeSumReduction : public shamrock::solvergraph::INode {
        u32 nvar;

        public:
        NodeSumReduction(u32 nvar) : nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeSumReduction"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shamrock::solvergraph

#undef NODE_EDGES
