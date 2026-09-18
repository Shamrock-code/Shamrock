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
 * @file ComputeGravWave.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Compute the gravitational wave quadrupole. Based on Toscani et. al. 2021.
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_positions)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_velocities)                                \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accelerations)                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_masses)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_accel_ext)                                 \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_pos)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_vel)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tvec>, central_acc)                                      \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, gw_prefactor)                                    \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, theta_gw)                                        \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, phi_gw)                                          \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IDataEdge<Tddq>, ddq)                                              \
    X_RW(shamrock::solvergraph::IDataEdge<Tddqxy>, ddq_xy)                                         \
    X_RW(shamrock::solvergraph::IDataEdge<Th>, hx)                                                 \
    X_RW(shamrock::solvergraph::IDataEdge<Th>, hp)

namespace shammodels::common::modules {

    template<class Tvec>
    class ComputeGravWave : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        public:
        using Tddq   = std::array<Tscal, 6>; // ddq: Matrix dot dot M
        using Tddqxy = std::array<Tscal, 9>;
        using Th     = std::array<Tscal, 4>;

        ComputeGravWave() = default;
        // explicit ComputeGravWave() : {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeGravWave"; }

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::common::modules

#undef NODE_EDGES
