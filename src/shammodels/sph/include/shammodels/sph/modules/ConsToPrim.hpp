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
 * @file ConsToPrim.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 @brief get primitive variables (rho, vel, internal energy) from conserved variables (rho*,
 momentum, entropy)
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <experimental/mdspan>

namespace shammodels::sph::modules {
    template<class Tvec, class SizeType, class Layout, class Accessor>
    class NodeConsToPrim : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        Tscal gamma;

        public:
        NodeConsToPrim(Tscal gamma) : gamma(gamma) {}

        using GcovEdge = std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout, Accessor>;

// X_RO(std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout, Accessor>, gcov)
#define NODE_CONS_TO_PRIM(X_RO, X_RW)                                                              \
    /* inputs */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhostar)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_momentum)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_K)                                        \
    X_RO(GcovEdge, gcon)                                                                           \
    X_RO(GcovEdge, gcov)                                                                           \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel)                                       \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_u)                                        \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_P)

        EXPAND_NODE_EDGES(NODE_CONS_TO_PRIM)
#undef NODE_CONS_TO_PRIM

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ConsToPrim"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules
