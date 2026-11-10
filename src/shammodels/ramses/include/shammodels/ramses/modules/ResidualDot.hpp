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
 * @file ResidualDot.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Squared L2 norm of a field, summed over all patches and ranks.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */
#include "shambase/aliases_int.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, spans_phi_res)                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, res_ddot)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute `res_ddot <- sum_i <res_i, res_i>` over the first `sizes * nvar` values of
     * every patch on every rank.
     *
     * `sizes` holds the object counts per patch, each object holding `nvar` values (e.g. the cells
     * of an AMR block). As in NodeSumReduction, the field may hold more values than summed (e.g.
     * ghost blocks stored after the patch blocks), but never less. The result is deterministic:
     * the patches are accumulated in increasing patch id order, then summed across ranks.
     */
    template<class T>
    class ResidualDot : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<T>;

        u32 nvar;

        public:
        ResidualDot(u32 nvar) : nvar(nvar) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ResidualDot"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
