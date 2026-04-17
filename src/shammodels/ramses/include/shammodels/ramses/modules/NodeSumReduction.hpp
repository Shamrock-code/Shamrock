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
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

#define NODE_SUMRED_EDGES(X_RO, X_RW)                                                              \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, spans_in)                                           \
    /* outputs*/                                                                                   \
    X_RW(shamrock::solvergraph::ScalarEdge<T>, out_scal)

namespace shammodels::basegodunov::modules {

    template<class T>
    class NodeSumReduction : public shamrock::solvergraph::INode {

        public:
        NodeSumReduction() {}

        EXPAND_NODE_EDGES(NODE_SUMRED_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeSumReduction"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules
