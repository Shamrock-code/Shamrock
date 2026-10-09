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
 * @file EulerTimeDerivativeDust.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Per cell Euler time derivatives of the dust primitive state
 *
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_vel_dust)                                  \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_grad_rho_dust)                             \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dx_v_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dy_v_dust)                                 \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dz_v_dust)                                 \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, spans_dt_rho_dust)                              \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, spans_dt_vel_dust)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Compute the Euler time derivatives of the dust primitive state per cell
     *
     * Dust is pressureless, so the velocity derivative carries only the
     * advection term. Same rationale as the gas counterpart: evaluate once per
     * cell instead of once per face side.
     */
    template<class Tvec>
    class NodeEulerTimeDerivativeDust : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        u32 ndust;

        public:
        NodeEulerTimeDerivativeDust(u32 block_size, u32 ndust)
            : block_size(block_size), ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "EulerTimeDerivativeDust"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
