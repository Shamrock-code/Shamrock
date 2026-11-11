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
 * @file AddExtForceBurkertProfile.hpp
 * @author Jérémy Fensch (jeremy.fensch@ens-lyon.fr): modified from a version from Noé Brucy
 (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula for the Burkert Profile.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(                                                                                          \
        shamrock::solvergraph::IFieldSpan<Tvec>,                                                   \
        spans_coordinates) /* center coordinates of each cell */                                   \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, constant_G) /* Grav Constant*/                  \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_a_ext)

namespace shammodels::basegodunov::modules {

    /// Add a gravitational acceleration g from a Burkert Profile to the external forces
    /// acceleration
    template<class Tvec>
    class AddExtForceBurkertProfile : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal rho_0;
        Tscal r_0;
        Tvec pos;

        public:
        AddExtForceBurkertProfile(u32 block_size, Tscal rho_0, Tscal r_0, Tvec pos)
            : block_size(block_size), rho_0(rho_0), r_0(r_0), pos(pos) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "AddExtForceBurkertProfile"; }

        inline virtual std::string _impl_get_tex() const { return "AddExtForceBurkertProfile"; }
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
