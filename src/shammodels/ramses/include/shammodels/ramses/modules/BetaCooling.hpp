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
 * @file BetaCooling.hpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Apply beta-cooling (cooling time proportional to the rotation time)
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt)                                             \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RO(                                                                                          \
        shamrock::solvergraph::IFieldSpan<Tvec>,                                                   \
        spans_coordinates) /* center coordinates of each cell */                                   \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)

namespace shammodels::basegodunov::modules {

    /// Apply beta-cooling,
    /// tcool = beta / omega
    template<class Tvec>
    class BetaCooling : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size; ///< number of cells per block
        Tscal beta;     ///< beta factor
        Tvec center;    ///< Center to compute omega

        public:
        BetaCooling(u32 block_size, Tscal beta, Tvec center)
            : block_size(block_size), beta(beta), center(center) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "BetaCooling"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
