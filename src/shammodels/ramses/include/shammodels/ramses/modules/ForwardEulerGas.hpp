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
 * @file ForwardEulerGas.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node performing the forward Euler time integration of the gas conservative state
 *
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
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_rhov)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rhoe)                                  \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_rhov)                                      \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rhoe)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Forward Euler step of the gas conservative variables, U <- U + dt * dU/dt.
     *
     * The state is updated in place.
     */
    template<class Tvec>
    class ForwardEulerGas : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;

        public:
        ForwardEulerGas(u32 block_size) : block_size(block_size) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ForwardEulerGas"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
