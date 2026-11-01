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
 * @brief Host-side (std::vector) forward Euler integration node.
 */

#include "shambase/stacktrace.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, dt)                                              \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<T>>, field_dt)                               \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<T>>, field)

namespace shamrock::solvergraph {

    /**
     * @brief Forward Euler integration of a host-side (std::vector) field.
     *
     * Performs field[i] += dt * field_dt[i] for every element of the field.
     *
     * @tparam T The value type stored in the field vector
     * @tparam Tscal The scalar type of the timestep dt (defaults to T)
     */
    template<class T, class Tscal = T>
    class ForwardEulerHost : public INode {

        public:
        ForwardEulerHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            __shamrock_stack_entry();

            auto edges = get_edges();

            Tscal dt                       = edges.dt.data;
            std::vector<T> &field          = edges.field.data;
            const std::vector<T> &field_dt = edges.field_dt.data;

            for (size_t i = 0; i < field.size(); i++) {
                field[i] += dt * field_dt[i];
            }
        }

        inline virtual std::string _impl_get_label() const { return "ForwardEulerHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shamrock::solvergraph

#undef NODE_EDGES
