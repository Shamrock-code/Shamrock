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
 * @brief
 *
 */
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

#define NODE_RESIDUALDOT_EDGES(X_RO, X_RW)                                                         \
    /*inputs*/                                                                                     \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldRefs<T>, spans_phi_res)                                      \
    /*outputs*/                                                                                    \
    X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, res_ddot)

namespace shammodels::basegodunov::modules {

    template<class T>
    class ResidualDot : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<T>;
        u32 block_size;

        public:
        ResidualDot(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_RESIDUALDOT_EDGES)
        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ResidualDot"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_RESIDUALDOT_EDGES
