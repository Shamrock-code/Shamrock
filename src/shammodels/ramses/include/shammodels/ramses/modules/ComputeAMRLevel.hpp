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
 * @file ComputeAMRLevel.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::ScalarsEdge<TgridVec>, level0_size)                                \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, block_min)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<TgridVec>, block_max)                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<TgridUint>, block_level)

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    class ComputeAMRLevel : public shamrock::solvergraph::INode {
        using Tgridscal = shambase::VecComponent<TgridVec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

        public:
        ComputeAMRLevel() {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline void _impl_reset_internal() {}

        inline virtual std::string _impl_get_label() const { return "ComputeAMRLevel"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
