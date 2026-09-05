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
 * @file ForwardEulerHost.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements a forward Euler integration step as a solver graph node, operating on
 * host-side std::vector data (e.g. MPI-replicated sink particles) rather than patch-distributed
 * GPU field spans.
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<T>>, time_derivative)                        \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<T>>, field)

namespace shammodels::common::modules {
    template<class T>
    class ForwardEulerHost : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<T>;

        public:
        ForwardEulerHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            Tscal dt = edges.dt.data;

            auto &field                 = edges.field.data;
            const std::vector<T> &deriv = edges.time_derivative.data;

            for (size_t i = 0; i < field.size(); i++) {
                field[i] = field[i] + dt * deriv[i];
            }
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEulerHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };
} // namespace shammodels::common::modules

#undef NODE_EDGES
