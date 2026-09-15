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
 * @file SinkSelfGravityHost.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Host-side pairwise gravitational self-interaction between sink particles.
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, G)                                               \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, epsilon)                                         \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, positions)                           \
    X_RO(shamrock::solvergraph::IDataEdge<std::vector<Tscal>>, masses)                             \
    X_RW(shamrock::solvergraph::IDataEdge<std::vector<Tvec>>, acc_ext)

namespace shammodels::sph::modules {

    /**
     * @brief Host-side pairwise (N^2) gravitational self-interaction between sink particles.
     *
     * Performs acc_ext[i] = -sum_j G*mass[j]*rij / (|rij|^3 + epsilon) for every sink.
     *
     * @tparam Tvec The sink position/velocity vector type
     */
    template<class Tvec>
    class SinkSelfGravityHost : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        SinkSelfGravityHost() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {
            __shamrock_stack_entry();

            auto edges = get_edges();

            Tscal G                        = edges.G.data;
            Tscal epsilon                  = edges.epsilon.data;
            const std::vector<Tvec> &pos   = edges.positions.data;
            const std::vector<Tscal> &mass = edges.masses.data;
            std::vector<Tvec> &acc_ext     = edges.acc_ext.data;

            for (size_t i = 0; i < pos.size(); i++) {
                Tvec sum{};
                for (size_t j = 0; j < pos.size(); j++) {
                    Tvec rij       = pos[i] - pos[j];
                    Tscal rij_scal = sycl::length(rij);
                    sum -= G * mass[j] * rij / (rij_scal * rij_scal * rij_scal + epsilon);
                }
                acc_ext[i] = sum;
            }
        }

        inline virtual std::string _impl_get_label() const { return "SinkSelfGravityHost"; }

        inline virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::sph::modules

#undef NODE_EDGES
