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
 * @file AddExtForcePointMassGravity.hpp
 * @author Noé Brucy (noe.brucy@ens-lyon.fr)
 * @brief Compute a gravitational acceleration at the center of each cell,
    using an analytical formula.
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
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

    /// Add the softened acceleration of a point mass,
    /// a = -G M (r - r0) / (|r - r0|^2 + eps^2)^{3/2}
    template<class Tvec>
    class AddExtForcePointMassGravity : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal mass;
        Tvec pos;
        Tscal epsilon_soft;

        public:
        AddExtForcePointMassGravity(u32 block_size, Tscal mass, Tvec pos, Tscal epsilon_soft)
            : block_size(block_size), mass(mass), pos(pos), epsilon_soft(epsilon_soft) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "AddExtForcePointMassGravity"; }

        inline virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
