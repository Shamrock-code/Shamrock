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
 * @file SlopeLimitedGradient.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

namespace shammodels::basegodunov::modules {

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, span_field)                                     \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, span_grad_field)

    template<class Tvec, class TgridVec>
    class SlopeLimitedScalarGradient : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        using SlopeMode = shammodels::basegodunov::SlopeMode;
        // alias so the template-argument comma does not split the NODE_EDGES macro arguments
        using AMRGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        u32 var_per_cell;
        SlopeMode mode;

        public:
        SlopeLimitedScalarGradient(u32 block_size, u32 var_per_cell, SlopeMode mode)
            : block_size(block_size), var_per_cell(var_per_cell), mode(mode) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SlopeLimitedScalarGradient"; };

        virtual std::string _impl_get_tex() const;
    };

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(AMRGraphEdge, cell_neigh_graph)                                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, span_field)                                      \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, span_dx_field)                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, span_dy_field)                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, span_dz_field)

    template<class Tvec, class TgridVec>
    class SlopeLimitedVectorGradient : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        using SlopeMode = shammodels::basegodunov::SlopeMode;
        // alias so the template-argument comma does not split the NODE_EDGES macro arguments
        using AMRGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        u32 var_per_cell;
        SlopeMode mode;

        public:
        SlopeLimitedVectorGradient(u32 block_size, u32 var_per_cell, SlopeMode mode)
            : block_size(block_size), var_per_cell(var_per_cell), mode(mode) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "SlopeLimitedVectorGradient"; };

        virtual std::string _impl_get_tex() const;
    };

#undef NODE_EDGES

} // namespace shammodels::basegodunov::modules
